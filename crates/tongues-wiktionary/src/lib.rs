//! Wiktionary pronunciation model-family data preparation.
//!
//! This family downloads the English Wiktionary MediaWiki XML dump and expands
//! extracted orthography/pronunciation pairs into multilingual seq2seq-style training rows.
//! The XML/wikitext extraction itself is intentionally stubbed until the parser
//! policy for Wiktionary pronunciation templates is implemented.

use std::collections::{BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use speech::data::spanish;
use tongues_core::Vocab;
use tongues_neural::{write_manifest, ModelArtifactManifest};
use unicode_normalization::UnicodeNormalization;

pub const FAMILY: &str = "wiktionary";
pub const ARCHITECTURE: &str = "wiktionary-pronunciation-seq2seq-scaffold";
pub const DEFAULT_DATASET_ID: &str = "enwiktionary-2026-06-01-v0";
pub const DEFAULT_DUMP_INDEX_URL: &str =
    "https://dumps.wikimedia.org/other/mediawiki_content_current/enwiktionary/2026-06-01/xml/bzip2/";
pub const DEFAULT_PIE_DATASET_ID: &str = "enwiktionary-pie-roots-2026-06-01-v0";
pub const DEFAULT_PIE_WIKIPEDIA_RAW_URL: &str =
    "https://en.wikipedia.org/w/index.php?title=Indo-European_vocabulary&action=raw";
const USER_AGENT: &str = "tongues-wiktionary/0.1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WiktionaryConfig {
    #[serde(default)]
    pub source_kind: WiktionarySourceKind,
    pub dataset_id: String,
    pub dump_index_url: String,
    #[serde(default)]
    pub dump_file_url: Option<String>,
    #[serde(default)]
    pub dump_path: Option<String>,
    #[serde(default)]
    pub wikipedia_raw_urls: Vec<String>,
    pub train_frac: f64,
    pub valid_frac: f64,
    pub seed: u64,
    pub languages: Vec<String>,
    #[serde(default = "default_train_task")]
    pub train_task: String,
    #[serde(default = "default_train_notations")]
    pub train_notations: Vec<String>,
    pub include_reverse: bool,
    pub include_language_guessing: bool,
    #[serde(default = "default_synthesize_spanish")]
    pub synthesize_spanish: bool,
    #[serde(default = "default_include_wiktionary_supplements")]
    pub include_wiktionary_supplements: bool,
    #[serde(default)]
    pub include_descendant_pairs: bool,
    #[serde(default)]
    pub max_pages: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WiktionarySourceKind {
    Pronunciation,
    PieEtymology,
}

impl Default for WiktionarySourceKind {
    fn default() -> Self {
        Self::Pronunciation
    }
}

impl Default for WiktionaryConfig {
    fn default() -> Self {
        Self {
            source_kind: WiktionarySourceKind::Pronunciation,
            dataset_id: DEFAULT_DATASET_ID.to_string(),
            dump_index_url: DEFAULT_DUMP_INDEX_URL.to_string(),
            dump_file_url: None,
            dump_path: None,
            wikipedia_raw_urls: Vec::new(),
            train_frac: 0.8,
            valid_frac: 0.1,
            seed: 42,
            languages: ["eng", "fra", "deu", "spa", "lat", "ell", "grc", "san"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            train_task: "all".to_string(),
            train_notations: default_train_notations(),
            include_reverse: true,
            include_language_guessing: true,
            synthesize_spanish: true,
            include_wiktionary_supplements: true,
            include_descendant_pairs: false,
            max_pages: None,
        }
    }
}

impl WiktionaryConfig {
    pub fn pie_etymology() -> Self {
        Self {
            source_kind: WiktionarySourceKind::PieEtymology,
            dataset_id: DEFAULT_PIE_DATASET_ID.to_string(),
            dump_index_url: DEFAULT_DUMP_INDEX_URL.to_string(),
            dump_file_url: None,
            dump_path: None,
            wikipedia_raw_urls: Vec::new(),
            train_frac: 0.8,
            valid_frac: 0.1,
            seed: 42,
            languages: pie_descendant_language_codes()
                .into_iter()
                .map(str::to_string)
                .collect(),
            train_task: "etymology-translation".to_string(),
            train_notations: Vec::new(),
            include_reverse: true,
            include_language_guessing: false,
            synthesize_spanish: false,
            include_wiktionary_supplements: false,
            include_descendant_pairs: false,
            max_pages: None,
        }
    }
}

fn default_train_task() -> String {
    "all".to_string()
}

fn default_train_notations() -> Vec<String> {
    ["phonemic", "phonetic"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn default_synthesize_spanish() -> bool {
    true
}

fn default_include_wiktionary_supplements() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationEntry {
    pub lang: String,
    pub wiktionary_lang: String,
    pub spelling: String,
    pub ipa: String,
    pub notation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accent: Option<String>,
    pub raw_template: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WiktionaryPattern {
    pub kind: String,
    pub lang: String,
    pub wiktionary_lang: String,
    pub spelling: String,
    pub values: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accent: Option<String>,
    pub raw_template: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupplementalTerm {
    pub domain: String,
    pub lang: String,
    pub wiktionary_lang: String,
    pub spelling: String,
    pub evidence: Vec<String>,
    pub has_pronunciation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PieEtymologyEntry {
    pub pie: String,
    pub lang: String,
    pub branch: String,
    pub descendant: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gloss: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingExample {
    pub task: WiktionaryTask,
    pub lang: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accent: Option<String>,
    pub input: String,
    pub output: String,
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WiktionaryTask {
    OrthographyToPhonology,
    PhonologyToOrthography,
    PhoneticRealization,
    EtymologyTranslation,
    PieToDescendant,
    DescendantToPie,
    DescendantToDescendant,
    AlignAudioText,
    NormalizeText,
    GuessLangFromOrthography,
    GuessLangFromPhonology,
    GuessLangFromOrthographyAndPhonology,
}

impl WiktionaryTask {
    pub fn token(self) -> &'static str {
        match self {
            Self::OrthographyToPhonology => "<task:orthography_to_phonology>",
            Self::PhonologyToOrthography => "<task:phonology_to_orthography>",
            Self::PhoneticRealization => "<task:phonetic_realization>",
            Self::EtymologyTranslation => "<task:etymology_translate>",
            Self::PieToDescendant => "<task:pie_to_descendant>",
            Self::DescendantToPie => "<task:descendant_to_pie>",
            Self::DescendantToDescendant => "<task:descendant_to_descendant>",
            Self::AlignAudioText => "<task:align>",
            Self::NormalizeText => "<task:normalize>",
            Self::GuessLangFromOrthography => "<task:guess_lang_from_orthography>",
            Self::GuessLangFromPhonology => "<task:guess_lang_from_phonology>",
            Self::GuessLangFromOrthographyAndPhonology => {
                "<task:guess_lang_from_orthography_and_phonology>"
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareReport {
    pub dump_path: String,
    pub extracted_patterns: usize,
    pub parsed_phonemes: usize,
    pub parsed_phones: usize,
    pub parsed_pie_roots: usize,
    pub train_examples: usize,
    pub valid_examples: usize,
    pub test_examples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareProgress {
    Stage {
        message: String,
    },
    Download {
        url: String,
        path: String,
        bytes: u64,
    },
    Parse {
        pages: usize,
        patterns: usize,
        phonemes: usize,
        phones: usize,
        pie_roots: usize,
    },
    Expand {
        rows: usize,
        examples: usize,
        path: Option<String>,
    },
    Write {
        path: String,
        rows: usize,
    },
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
    prepare_dataset_with_progress(out, cache_dir, config, |_| {})
}

pub fn prepare_dataset_with_progress(
    out: &Path,
    cache_dir: &Path,
    config: &WiktionaryConfig,
    mut progress: impl FnMut(PrepareProgress),
) -> Result<PrepareReport> {
    progress(PrepareProgress::Stage {
        message: format!("Creating output/cache directories: {}", out.display()),
    });
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::create_dir_all(cache_dir).with_context(|| format!("creating {}", cache_dir.display()))?;

    if config.source_kind == WiktionarySourceKind::PieEtymology {
        return prepare_pie_dataset(out, cache_dir, config, &mut progress);
    }

    let dump_path = resolve_dump_path_with_progress(cache_dir, config, &mut progress)?;
    let extracted = parse_dump_with_progress(&dump_path, config, &mut progress)?;
    let phonemes = extracted.phonemes;
    let phones = extracted.phones;
    progress(PrepareProgress::Stage {
        message: format!(
            "Expanding {} phoneme and {} phone rows into training examples",
            phonemes.len(),
            phones.len()
        ),
    });
    let expanded_path = out.join("expanded.jsonl.part");
    progress(PrepareProgress::Stage {
        message: format!(
            "Writing expanded training examples to {}",
            expanded_path.display()
        ),
    });
    let mut expanded_file = BufWriter::new(
        File::create(&expanded_path)
            .with_context(|| format!("creating {}", expanded_path.display()))?,
    );
    let mut examples = Vec::new();
    let entries = phonemes
        .iter()
        .chain(phones.iter())
        .cloned()
        .collect::<Vec<_>>();
    expand_training_examples_to(
        &entries,
        config,
        &mut progress,
        Some(&expanded_path),
        |example| {
            writeln!(expanded_file, "{}", serde_json::to_string(&example)?)?;
            examples.push(example);
            Ok(())
        },
    )?;
    expanded_file
        .flush()
        .with_context(|| format!("flushing {}", expanded_path.display()))?;
    progress(PrepareProgress::Stage {
        message: format!(
            "Splitting {} examples into train/valid/test",
            examples.len()
        ),
    });
    let (train, valid, test) =
        split_examples(examples, config.train_frac, config.valid_frac, config.seed);

    write_jsonl_with_progress(&out.join("train.jsonl"), &train, &mut progress)?;
    write_jsonl_with_progress(&out.join("valid.jsonl"), &valid, &mut progress)?;
    write_jsonl_with_progress(&out.join("test.jsonl"), &test, &mut progress)?;
    write_jsonl_with_progress(
        &out.join("patterns.jsonl"),
        &extracted.patterns,
        &mut progress,
    )?;
    write_jsonl_with_progress(&out.join("phonemes.jsonl"), &phonemes, &mut progress)?;
    write_jsonl_with_progress(&out.join("phones.jsonl"), &phones, &mut progress)?;
    write_jsonl_with_progress(
        &out.join("supplemental_terms.jsonl"),
        &extracted.supplemental_terms,
        &mut progress,
    )?;
    progress(PrepareProgress::Stage {
        message: "Building vocabulary".to_string(),
    });
    write_vocab(out, [&train[..], &valid[..], &test[..]].concat().as_slice())?;
    progress(PrepareProgress::Write {
        path: out.join("vocab.json").display().to_string(),
        rows: train.len() + valid.len() + test.len(),
    });
    fs::write(
        out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(out.join("README.md"), dataset_readme(config, &dump_path))?;
    fs::remove_file(&expanded_path)
        .with_context(|| format!("removing {}", expanded_path.display()))?;

    Ok(PrepareReport {
        dump_path: dump_path.display().to_string(),
        extracted_patterns: extracted.patterns.len(),
        parsed_phonemes: phonemes.len(),
        parsed_phones: phones.len(),
        parsed_pie_roots: 0,
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
    })
}

fn prepare_pie_dataset(
    out: &Path,
    cache_dir: &Path,
    config: &WiktionaryConfig,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<PrepareReport> {
    let dump_path = resolve_dump_path_with_progress(cache_dir, config, progress)?;
    let extracted = parse_dump_with_progress(&dump_path, config, progress)?;
    let mut roots = extracted.pie_roots;
    let mut source_paths = vec![dump_path];
    let wikipedia_paths =
        resolve_wikipedia_source_paths_with_progress(cache_dir, config, progress)?;
    source_paths.extend(wikipedia_paths.iter().cloned());
    for path in &wikipedia_paths {
        progress(PrepareProgress::Stage {
            message: format!("Reading supplemental source {}", path.display()),
        });
        let raw =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        roots.extend(extract_pie_etymology_entries(&raw, config));
    }
    progress(PrepareProgress::Stage {
        message: format!("Sorting and deduplicating {} PIE root rows", roots.len()),
    });
    roots.sort_by(|a, b| {
        (&a.pie, &a.lang, &a.branch, &a.descendant).cmp(&(
            &b.pie,
            &b.lang,
            &b.branch,
            &b.descendant,
        ))
    });
    roots.dedup_by(|a, b| {
        a.pie == b.pie && a.lang == b.lang && a.branch == b.branch && a.descendant == b.descendant
    });

    progress(PrepareProgress::Stage {
        message: format!(
            "Expanding {} PIE root rows into etymology examples",
            roots.len()
        ),
    });
    let examples = expand_pie_training_examples(&roots, config);
    progress(PrepareProgress::Stage {
        message: format!(
            "Splitting {} examples into train/valid/test",
            examples.len()
        ),
    });
    let (train, valid, test) =
        split_examples(examples, config.train_frac, config.valid_frac, config.seed);

    write_jsonl_with_progress(&out.join("train.jsonl"), &train, progress)?;
    write_jsonl_with_progress(&out.join("valid.jsonl"), &valid, progress)?;
    write_jsonl_with_progress(&out.join("test.jsonl"), &test, progress)?;
    write_jsonl_with_progress(&out.join("pie_roots.jsonl"), &roots, progress)?;
    progress(PrepareProgress::Stage {
        message: "Building vocabulary".to_string(),
    });
    write_vocab(out, [&train[..], &valid[..], &test[..]].concat().as_slice())?;
    progress(PrepareProgress::Write {
        path: out.join("vocab.json").display().to_string(),
        rows: train.len() + valid.len() + test.len(),
    });
    fs::write(
        out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("README.md"),
        pie_dataset_readme(config, &source_paths),
    )?;

    Ok(PrepareReport {
        dump_path: source_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        extracted_patterns: roots.len(),
        parsed_phonemes: 0,
        parsed_phones: 0,
        parsed_pie_roots: roots.len(),
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
    })
}

fn resolve_wikipedia_source_paths_with_progress(
    cache_dir: &Path,
    config: &WiktionaryConfig,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<Vec<PathBuf>> {
    if let Some(path) = &config.dump_path {
        return Ok(vec![PathBuf::from(path)]);
    }
    let urls = config.wikipedia_raw_urls.clone();
    let mut paths = Vec::new();
    for (index, url) in urls.iter().enumerate() {
        let filename = wikipedia_cache_filename(url, index);
        let path = cache_dir.join(filename);
        if !path.exists() || path.metadata()?.len() == 0 {
            download_to_file_with_progress(url, &path, progress)?;
        } else {
            progress(PrepareProgress::Stage {
                message: format!("Using cached supplemental source {}", path.display()),
            });
        }
        paths.push(path);
    }
    Ok(paths)
}

fn wikipedia_cache_filename(url: &str, index: usize) -> String {
    let title = url
        .split("title=")
        .nth(1)
        .and_then(|tail| tail.split('&').next())
        .unwrap_or("wikipedia-pie-source")
        .replace("%20", "_")
        .replace(['/', '\\', ':', '?', '&', '='], "_");
    format!("{index:02}-{title}.wiki")
}

pub fn resolve_dump_path(cache_dir: &Path, config: &WiktionaryConfig) -> Result<PathBuf> {
    resolve_dump_path_with_progress(cache_dir, config, &mut |_| {})
}

fn resolve_dump_path_with_progress(
    cache_dir: &Path,
    config: &WiktionaryConfig,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<PathBuf> {
    if let Some(path) = &config.dump_path {
        progress(PrepareProgress::Stage {
            message: format!("Using configured dump {}", path),
        });
        return Ok(PathBuf::from(path));
    }
    download_dump_with_progress(cache_dir, config, progress)
}

pub fn download_dump(cache_dir: &Path, config: &WiktionaryConfig) -> Result<PathBuf> {
    download_dump_with_progress(cache_dir, config, &mut |_| {})
}

fn download_dump_with_progress(
    cache_dir: &Path,
    config: &WiktionaryConfig,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<PathBuf> {
    progress(PrepareProgress::Stage {
        message: "Resolving Wiktionary dump URL".to_string(),
    });
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
        progress(PrepareProgress::Stage {
            message: format!("Using cached dump {}", path.display()),
        });
        return Ok(path);
    }
    download_to_file_with_progress(&dump_url, &path, progress)?;
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

fn download_to_file_with_progress(
    url: &str,
    path: &Path,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<()> {
    let part_path = path.with_extension("part");
    progress(PrepareProgress::Stage {
        message: format!("Downloading {url}"),
    });
    let response = ureq::get(url)
        .header("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let mut body = response.into_body();
    let mut reader = body.as_reader();
    let mut file =
        File::create(&part_path).with_context(|| format!("creating {}", part_path.display()))?;
    let mut buffer = [0_u8; 1024 * 64];
    let mut downloaded = 0_u64;
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        downloaded += n as u64;
        if downloaded < 1024 * 1024 || downloaded % (8 * 1024 * 1024) < n as u64 {
            progress(PrepareProgress::Download {
                url: url.to_string(),
                path: path.display().to_string(),
                bytes: downloaded,
            });
        }
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

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ExtractedWiktionaryData {
    pub patterns: Vec<WiktionaryPattern>,
    pub phonemes: Vec<PronunciationEntry>,
    pub phones: Vec<PronunciationEntry>,
    pub supplemental_terms: Vec<SupplementalTerm>,
    pub pie_roots: Vec<PieEtymologyEntry>,
}

pub fn parse_dump(dump_path: &Path, config: &WiktionaryConfig) -> Result<ExtractedWiktionaryData> {
    parse_dump_with_progress(dump_path, config, &mut |_| {})
}

fn parse_dump_with_progress(
    dump_path: &Path,
    config: &WiktionaryConfig,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<ExtractedWiktionaryData> {
    progress(PrepareProgress::Stage {
        message: format!("Opening dump {}", dump_path.display()),
    });
    let file = File::open(dump_path).with_context(|| format!("opening {}", dump_path.display()))?;
    if dump_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension == "bz2")
    {
        progress(PrepareProgress::Stage {
            message: format!("Decompressing and parsing {}", dump_path.display()),
        });
        let decoder = BzDecoder::new(file);
        let reader = BufReader::with_capacity(1024 * 1024, decoder);
        parse_xml_pages_with_progress(reader, config, progress)
    } else {
        progress(PrepareProgress::Stage {
            message: format!("Parsing {}", dump_path.display()),
        });
        let reader = BufReader::with_capacity(1024 * 1024, file);
        parse_xml_pages_with_progress(reader, config, progress)
    }
}

fn parse_xml_pages_with_progress<R: BufRead>(
    reader: R,
    config: &WiktionaryConfig,
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<ExtractedWiktionaryData> {
    let mut data = ExtractedWiktionaryData::default();
    let mut title = String::new();
    let mut text = String::new();
    let mut in_text = false;
    let mut pages_seen = 0_usize;

    for line in reader.lines() {
        let line = line?;
        if !in_text {
            if let Some(value) = xml_tag_value(&line, "title") {
                title = decode_xml_entities(value);
            }
            if let Some(start) = line.find("<text") {
                in_text = true;
                if let Some(gt) = line[start..].find('>') {
                    let after = &line[start + gt + 1..];
                    if let Some(end) = after.find("</text>") {
                        text.push_str(&decode_xml_entities(&after[..end]));
                        data.extend(extract_page_data(&title, &text, config));
                        text.clear();
                        in_text = false;
                        pages_seen += 1;
                        maybe_report_parse_progress(progress, pages_seen, &data);
                        if config.max_pages.is_some_and(|max| pages_seen >= max) {
                            break;
                        }
                    } else {
                        text.push_str(after);
                        text.push('\n');
                    }
                }
            }
        } else if let Some(end) = line.find("</text>") {
            text.push_str(&decode_xml_entities(&line[..end]));
            data.extend(extract_page_data(&title, &text, config));
            text.clear();
            in_text = false;
            pages_seen += 1;
            maybe_report_parse_progress(progress, pages_seen, &data);
            if config.max_pages.is_some_and(|max| pages_seen >= max) {
                break;
            }
        } else {
            text.push_str(&decode_xml_entities(&line));
            text.push('\n');
        }
    }

    progress(PrepareProgress::Parse {
        pages: pages_seen,
        patterns: data.patterns.len(),
        phonemes: data.phonemes.len(),
        phones: data.phones.len(),
        pie_roots: data.pie_roots.len(),
    });

    Ok(data)
}

fn maybe_report_parse_progress(
    progress: &mut impl FnMut(PrepareProgress),
    pages_seen: usize,
    data: &ExtractedWiktionaryData,
) {
    if pages_seen <= 10 || pages_seen % 1_000 == 0 {
        progress(PrepareProgress::Parse {
            pages: pages_seen,
            patterns: data.patterns.len(),
            phonemes: data.phonemes.len(),
            phones: data.phones.len(),
            pie_roots: data.pie_roots.len(),
        });
    }
}

impl ExtractedWiktionaryData {
    fn extend(&mut self, other: ExtractedWiktionaryData) {
        self.patterns.extend(other.patterns);
        self.phonemes.extend(other.phonemes);
        self.phones.extend(other.phones);
        self.supplemental_terms.extend(other.supplemental_terms);
        self.pie_roots.extend(other.pie_roots);
    }
}

fn xml_tag_value<'a>(line: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = line.find(&open)? + open.len();
    let end = line[start..].find(&close)? + start;
    Some(&line[start..end])
}

pub fn extract_pronunciations(
    spelling: &str,
    wikitext: &str,
    config: &WiktionaryConfig,
) -> Vec<PronunciationEntry> {
    extract_page_data(spelling, wikitext, config).phonemes
}

pub fn extract_page_data(
    spelling: &str,
    wikitext: &str,
    config: &WiktionaryConfig,
) -> ExtractedWiktionaryData {
    if config.source_kind == WiktionarySourceKind::PieEtymology {
        return ExtractedWiktionaryData {
            pie_roots: extract_wiktionary_pie_etymology_entries(spelling, wikitext, config),
            ..ExtractedWiktionaryData::default()
        };
    }

    if spelling.is_empty() || spelling.contains(':') {
        return ExtractedWiktionaryData::default();
    }

    let allowed = allowed_wiktionary_langs(config);
    let mut data = ExtractedWiktionaryData::default();
    let mut seen = HashSet::new();
    let supplements = if config.include_wiktionary_supplements {
        classify_supplemental_terms(spelling, wikitext, &allowed)
    } else {
        Vec::new()
    };
    for template in find_named_templates(wikitext, &["IPA", "audio", "homophones", "rhymes"]) {
        let params = split_template_params(template);
        if params.len() < 2 {
            continue;
        }
        let kind = params[0].trim();
        let wiktionary_lang = params[1].trim();
        if !allowed.contains(wiktionary_lang) {
            continue;
        }
        let lang = match iso3_from_wiktionary_lang(wiktionary_lang) {
            Some(lang) => lang.to_string(),
            None => continue,
        };
        let accent =
            template_named_param(&params, "a").or_else(|| template_named_param(&params, "aa"));
        let values = params
            .iter()
            .skip(2)
            .map(|value| value.trim())
            .filter(|value| !value.is_empty() && !value.contains('='))
            .map(str::to_string)
            .collect::<Vec<_>>();
        if !values.is_empty() {
            data.patterns.push(WiktionaryPattern {
                kind: kind.to_ascii_lowercase(),
                lang: lang.clone(),
                wiktionary_lang: wiktionary_lang.to_string(),
                spelling: spelling.to_string(),
                values: values.clone(),
                accent: accent.clone(),
                raw_template: format!("{{{{{template}}}}}"),
            });
        }
        if !kind.eq_ignore_ascii_case("IPA") {
            continue;
        }
        for value in values {
            let value = value.trim();
            let Some(notation) = ipa_notation(value) else {
                continue;
            };
            let key = format!("{lang}\t{spelling}\t{value}");
            if seen.insert(key) {
                let entry = PronunciationEntry {
                    lang: lang.clone(),
                    wiktionary_lang: wiktionary_lang.to_string(),
                    spelling: spelling.to_string(),
                    ipa: value.to_string(),
                    notation: notation.to_string(),
                    accent: accent.clone(),
                    raw_template: format!("{{{{{template}}}}}"),
                };
                match notation {
                    "phonemic" => data.phonemes.push(entry),
                    "phonetic" => data.phones.push(entry),
                    _ => {}
                }
            }
        }
    }
    if config.synthesize_spanish
        && allowed.contains("es")
        && has_language_section(wikitext, "Spanish")
    {
        for (variety, ipa) in spanish::synthetic_pronunciations(spelling) {
            let key = format!("spa\t{spelling}\t{ipa}");
            if seen.insert(key) {
                data.phonemes.push(PronunciationEntry {
                    lang: "spa".to_string(),
                    wiktionary_lang: "es".to_string(),
                    spelling: spelling.to_string(),
                    ipa,
                    notation: "phonemic".to_string(),
                    accent: Some(variety.accent_tag().to_string()),
                    raw_template: format!(
                        "{{{{synthetic-spanish|{}|{}}}}}",
                        variety.id(),
                        spelling
                    ),
                });
            }
        }
    }
    if !supplements.is_empty() {
        let has_pronunciation = !data.phonemes.is_empty() || !data.phones.is_empty();
        data.supplemental_terms = supplements
            .iter()
            .map(|supplement| SupplementalTerm {
                domain: supplement.domain.to_string(),
                lang: supplement.lang.to_string(),
                wiktionary_lang: supplement.wiktionary_lang.to_string(),
                spelling: spelling.to_string(),
                evidence: supplement.evidence.clone(),
                has_pronunciation,
            })
            .collect();
        append_supplemental_pronunciation_rows(&mut data, &supplements, &mut seen);
    }
    data
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SupplementalTermMatch {
    domain: &'static str,
    accent: &'static str,
    lang: &'static str,
    wiktionary_lang: &'static str,
    evidence: Vec<String>,
}

fn classify_supplemental_terms(
    spelling: &str,
    wikitext: &str,
    allowed: &BTreeSet<&str>,
) -> Vec<SupplementalTermMatch> {
    let mut matches = Vec::new();
    let lower = wikitext.to_ascii_lowercase();

    if allowed.contains("en")
        && has_language_section(wikitext, "English")
        && spelling.chars().next().is_some_and(char::is_uppercase)
        && contains_any(
            &lower,
            &[
                "derived from ancient greek",
                "from ancient greek",
                "greek given names",
                "greek surnames",
                "category:english terms derived from ancient greek",
                "{{given name",
                "{{surname",
            ],
        )
    {
        matches.push(SupplementalTermMatch {
            domain: "english-greek-name",
            accent: "GreekName",
            lang: "eng",
            wiktionary_lang: "en",
            evidence: supplemental_evidence(
                &lower,
                &[
                    "derived from ancient greek",
                    "from ancient greek",
                    "greek given names",
                    "greek surnames",
                    "{{given name",
                    "{{surname",
                ],
            ),
        });
    }

    if allowed.contains("la") && has_language_section(wikitext, "Latin") {
        matches.push(SupplementalTermMatch {
            domain: "latin",
            accent: "Latin",
            lang: "lat",
            wiktionary_lang: "la",
            evidence: vec!["==Latin==".to_string()],
        });
    }

    if (allowed.contains("la") || allowed.contains("en"))
        && contains_any(
            &lower,
            &[
                "new latin",
                "neo-latin",
                "scientific name",
                "taxonomic name",
                "{{taxon",
                "{{species",
                "{{taxlink",
                "category:translingual taxonomic names",
                "category:species",
            ],
        )
    {
        let latin = has_language_section(wikitext, "Latin");
        matches.push(SupplementalTermMatch {
            domain: "neo-latin-scientific",
            accent: "NeoLatinScientific",
            lang: if latin { "lat" } else { "eng" },
            wiktionary_lang: if latin { "la" } else { "en" },
            evidence: supplemental_evidence(
                &lower,
                &[
                    "new latin",
                    "neo-latin",
                    "scientific name",
                    "taxonomic name",
                    "{{taxon",
                    "{{species",
                    "{{taxlink",
                ],
            ),
        });
    }

    if (allowed.contains("la") || allowed.contains("en"))
        && contains_any(
            &lower,
            &[
                "legal latin",
                "category:legal latin",
                "category:english legal terms",
                "category:latin legal terms",
                "{{lb|en|law",
                "{{lb|la|law",
                "{{legal",
            ],
        )
    {
        let latin = has_language_section(wikitext, "Latin");
        matches.push(SupplementalTermMatch {
            domain: "legal-latin",
            accent: "LegalLatin",
            lang: if latin { "lat" } else { "eng" },
            wiktionary_lang: if latin { "la" } else { "en" },
            evidence: supplemental_evidence(
                &lower,
                &[
                    "legal latin",
                    "category:legal latin",
                    "category:english legal terms",
                    "category:latin legal terms",
                    "{{lb|en|law",
                    "{{lb|la|law",
                    "{{legal",
                ],
            ),
        });
    }

    matches
}

fn append_supplemental_pronunciation_rows(
    data: &mut ExtractedWiktionaryData,
    supplements: &[SupplementalTermMatch],
    seen: &mut HashSet<String>,
) {
    let phonemes = data.phonemes.clone();
    let phones = data.phones.clone();
    for supplement in supplements {
        for entry in phonemes.iter().filter(|entry| {
            entry.lang == supplement.lang && entry.wiktionary_lang == supplement.wiktionary_lang
        }) {
            append_supplemental_pronunciation_row(
                &mut data.phonemes,
                entry,
                supplement,
                "phonemes",
                seen,
            );
        }
        for entry in phones.iter().filter(|entry| {
            entry.lang == supplement.lang && entry.wiktionary_lang == supplement.wiktionary_lang
        }) {
            append_supplemental_pronunciation_row(
                &mut data.phones,
                entry,
                supplement,
                "phones",
                seen,
            );
        }
    }
}

fn append_supplemental_pronunciation_row(
    rows: &mut Vec<PronunciationEntry>,
    entry: &PronunciationEntry,
    supplement: &SupplementalTermMatch,
    kind: &str,
    seen: &mut HashSet<String>,
) {
    let key = format!(
        "{}\t{}\t{}\t{}",
        entry.lang, entry.spelling, entry.ipa, supplement.domain
    );
    if seen.insert(key) {
        let mut row = entry.clone();
        row.accent = Some(supplement.accent.to_string());
        row.raw_template = format!(
            "{{{{wiktionary-supplement|{}|{}|{}}}}}",
            kind, supplement.domain, entry.spelling
        );
        rows.push(row);
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn supplemental_evidence(haystack: &str, needles: &[&str]) -> Vec<String> {
    needles
        .iter()
        .filter(|needle| haystack.contains(**needle))
        .map(|needle| (*needle).to_string())
        .collect()
}

fn has_language_section(wikitext: &str, language: &str) -> bool {
    wikitext.lines().any(|line| {
        let trimmed = line.trim();
        trimmed.starts_with("==")
            && trimmed.ends_with("==")
            && trimmed.trim_matches('=').trim() == language
    })
}

pub fn extract_pie_etymology_entries(
    wikitext: &str,
    config: &WiktionaryConfig,
) -> Vec<PieEtymologyEntry> {
    let allowed: BTreeSet<&str> = config.languages.iter().map(String::as_str).collect();
    let mut entries = Vec::new();
    for table in find_wikitables(wikitext) {
        let headers = parse_table_headers(table);
        if headers
            .first()
            .is_none_or(|header| pie_column_code(header).is_none())
        {
            continue;
        }
        for row in parse_table_rows(table) {
            if row.len() < 2 {
                continue;
            }
            let pie = clean_wikitext_cell(&row[0]);
            if pie.is_empty() {
                continue;
            }
            let gloss = extract_quoted_gloss(&row[0]).or_else(|| extract_quoted_gloss(&pie));
            for (index, cell) in row.iter().enumerate().skip(1) {
                let Some(header) = headers.get(index) else {
                    continue;
                };
                let Some((lang, branch)) = descendant_column(header) else {
                    continue;
                };
                if !allowed.is_empty()
                    && !allowed.contains(lang)
                    && !allowed.contains(branch)
                    && !allowed.contains("ine-pro")
                {
                    continue;
                }
                let descendant = clean_wikitext_cell(cell);
                if !is_valid_pie_form(&pie) || !is_valid_descendant_form(&descendant) {
                    continue;
                }
                entries.push(PieEtymologyEntry {
                    pie: pie.clone(),
                    lang: lang.to_string(),
                    branch: branch.to_string(),
                    descendant,
                    gloss: gloss.clone(),
                    source: "wikipedia:Indo-European vocabulary".to_string(),
                });
            }
        }
    }
    entries
}

pub fn extract_wiktionary_pie_etymology_entries(
    spelling: &str,
    wikitext: &str,
    config: &WiktionaryConfig,
) -> Vec<PieEtymologyEntry> {
    if !is_pie_etymology_page_title(spelling) {
        return Vec::new();
    }
    let allowed: BTreeSet<&str> = config.languages.iter().map(String::as_str).collect();
    let page_form = wiktionary_page_form(spelling);
    let mut entries = Vec::new();
    let initial_lang = wiktionary_lang_from_heading(wikitext.lines().next().unwrap_or(""));
    let mut current_pie = if initial_lang.as_deref() == Some("ine-pro") {
        Some(page_form.clone())
    } else {
        None
    };

    for line in wikitext.lines() {
        if let Some(lang) = wiktionary_lang_from_heading(line) {
            current_pie = (lang == "ine-pro").then(|| page_form.clone());
        }

        for template in
            find_named_templates(line, &["root", "der", "inh", "desc", "desctree", "etymon"])
        {
            let params = split_template_params(template);
            if params.is_empty() {
                continue;
            }
            let name = params[0].trim().to_ascii_lowercase();
            match name.as_str() {
                "root" => {
                    if params.get(2).is_some_and(|lang| lang.trim() == "ine-pro") {
                        if let (Some(lang), Some(pie)) = (params.get(1), params.get(3)) {
                            push_pie_entry(
                                &mut entries,
                                &allowed,
                                clean_template_form(pie),
                                lang.trim(),
                                &page_form,
                                template_named_param(&params, "t"),
                                "enwiktionary:root-template",
                            );
                        }
                    }
                }
                "der" | "inh" => {
                    if params.get(2).is_some_and(|lang| lang.trim() == "ine-pro") {
                        if let (Some(lang), Some(pie)) = (params.get(1), params.get(3)) {
                            push_pie_entry(
                                &mut entries,
                                &allowed,
                                clean_template_form(pie),
                                lang.trim(),
                                &page_form,
                                template_named_param(&params, "t"),
                                "enwiktionary:etymology-template",
                            );
                        }
                    }
                }
                "etymon" => {
                    if params.get(1).is_some_and(|lang| lang.trim() == "ine-pro") {
                        current_pie = Some(page_form.clone());
                    }
                }
                "desc" | "desctree" => {
                    let Some(pie) = current_pie.as_deref() else {
                        continue;
                    };
                    let Some(lang) = params.get(1).map(|lang| lang.trim()) else {
                        continue;
                    };
                    let descendant = template_form_param(&params)
                        .or_else(|| template_named_param(&params, "alt"))
                        .or_else(|| template_named_param(&params, "alt1"))
                        .unwrap_or_default();
                    push_pie_entry(
                        &mut entries,
                        &allowed,
                        pie.to_string(),
                        lang,
                        &clean_template_form(&descendant),
                        template_named_param(&params, "t"),
                        "enwiktionary:desc-template",
                    );
                }
                _ => {}
            }
        }
    }

    entries.sort_by(|a, b| {
        (&a.pie, &a.lang, &a.descendant, &a.source).cmp(&(
            &b.pie,
            &b.lang,
            &b.descendant,
            &b.source,
        ))
    });
    entries.dedup_by(|a, b| {
        a.pie == b.pie && a.lang == b.lang && a.descendant == b.descendant && a.source == b.source
    });
    entries
}

fn is_pie_etymology_page_title(title: &str) -> bool {
    if title.trim().is_empty() {
        return false;
    }
    !title.contains(':') || title.starts_with("Reconstruction:Proto-Indo-European/")
}

fn push_pie_entry(
    entries: &mut Vec<PieEtymologyEntry>,
    allowed: &BTreeSet<&str>,
    pie: String,
    lang: &str,
    descendant: &str,
    gloss: Option<String>,
    source: &str,
) {
    let pie = clean_template_form(&pie);
    let descendant = clean_template_form(descendant);
    if !is_valid_pie_form(&pie)
        || !is_valid_descendant_form(&descendant)
        || lang.is_empty()
        || lang == "ine-pro"
    {
        return;
    }
    let branch = pie_branch_for_wiktionary_lang(lang);
    if !allowed.is_empty() && !allowed.contains(lang) && !allowed.contains(branch) {
        return;
    }
    entries.push(PieEtymologyEntry {
        pie,
        lang: lang.to_string(),
        branch: branch.to_string(),
        descendant,
        gloss: gloss.map(|value| clean_template_form(&value)),
        source: source.to_string(),
    });
}

fn is_valid_pie_form(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && trimmed != "*"
        && trimmed != "-"
        && trimmed.starts_with('*')
        && !trimmed.contains(':')
}

fn is_valid_descendant_form(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "-" || trimmed == "*" {
        return false;
    }
    let lowered = trimmed.to_ascii_lowercase();
    !matches!(
        lowered.as_str(),
        "inherited from pie root"
            | "derived from pie root"
            | "borrowed from pie root"
            | "see desc"
            | "derived terms"
    ) && !trimmed.contains("Category:")
        && !trimmed.contains("User:")
}

fn template_form_param(params: &[String]) -> Option<String> {
    params
        .get(2)
        .filter(|value| !value.trim().is_empty() && !value.contains('='))
        .cloned()
        .or_else(|| params.get(3).filter(|value| !value.contains('=')).cloned())
}

fn wiktionary_page_form(title: &str) -> String {
    let leaf = title
        .rsplit('/')
        .next()
        .unwrap_or(title)
        .trim()
        .trim_start_matches("Reconstruction:");
    let form = clean_template_form(leaf);
    if title.contains("Proto-Indo-European/") && !form.starts_with('*') {
        format!("*{form}")
    } else {
        form
    }
}

fn clean_template_form(value: &str) -> String {
    clean_wikitext_cell(value)
        .trim_matches(|ch: char| matches!(ch, '[' | ']' | '{' | '}' | '|'))
        .trim()
        .to_string()
}

fn wiktionary_lang_from_heading(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if !(trimmed.starts_with("==") && trimmed.ends_with("==")) {
        return None;
    }
    let level = trimmed.chars().take_while(|ch| *ch == '=').count();
    if level != 2 {
        return None;
    }
    let heading = trimmed.trim_matches('=').trim();
    Some(
        match heading {
            "English" => "en",
            "Middle English" => "enm",
            "Old English" => "ang",
            "Old Dutch" => "odt",
            "Old Saxon" => "osx",
            "Old Norse" => "non",
            "German" => "de",
            "Dutch" => "nl",
            "Proto-Indo-European" => "ine-pro",
            "Proto-Celtic" => "cel-pro",
            "Proto-Germanic" => "gem-pro",
            "Proto-West Germanic" => "gmw-pro",
            "Proto-Brythonic" => "cel-bry-pro",
            "Proto-Italic" => "itc-pro",
            "Latin" => "la",
            "Ancient Greek" => "grc",
            "Sanskrit" => "sa",
            "Avestan" => "ae",
            "Old Persian" => "peo",
            "Lithuanian" => "lt",
            "Latvian" => "lv",
            "Old Church Slavonic" => "cu",
            "Armenian" => "hy",
            "Albanian" => "sq",
            "Hittite" => "hit",
            "Tocharian A" => "xto",
            "Tocharian B" => "txb",
            _ => return None,
        }
        .to_string(),
    )
}

fn pie_branch_for_wiktionary_lang(lang: &str) -> &'static str {
    match lang {
        "ine-pro" => "pie",
        "en" | "enm" | "ang" | "sco" | "de" | "nl" | "odt" | "osx" | "non" | "is" | "da" | "sv"
        | "no" | "nb" | "nn" | "fy" | "stq" | "nds" | "nds-de" | "nds-nl" | "gem-pro"
        | "gmw-pro" | "gmq-pro" | "gmw-cfr" | "gmw-msc" | "gml" | "got" => "germanic",
        "la" | "itc-pro" | "xum" | "osc" | "it" | "fr" | "es" | "pt" | "ro" | "pro" => "italic",
        "grc" | "el" => "hellenic",
        "sa" | "inc-pro" | "pi" | "hi" | "ur" | "bn" | "pa" | "mr" | "ne" => "indo-aryan",
        "ira-pro" | "ira" | "ae" | "peo" | "pal" | "fa" | "ku" | "ps" | "os" => "iranian",
        "sla-pro" | "ine-bsl-pro" | "cu" | "ru" | "uk" | "pl" | "cs" | "sk" | "bg" | "sh"
        | "sl" => "slavic",
        "bat-pro" | "lt" | "lv" | "prg" => "baltic",
        "cel-pro" | "cel-bry-pro" | "cel-gau" | "sga" | "mga" | "ga" | "cy" | "wlm" | "owl"
        | "br" | "kw" => "celtic",
        "hy" | "xcl" => "armenian",
        "sq" | "sqj-pro" => "albanian",
        "txb" | "xto" | "txh" => "tocharian",
        "hit" | "luw" | "xlu" | "xlc" | "lyd" | "xld" => "anatolian",
        _ if lang.ends_with("-pro") => "proto-indo-european-descendant",
        _ => "indo-european-descendant",
    }
}

fn find_wikitables(wikitext: &str) -> Vec<&str> {
    let mut tables = Vec::new();
    let mut offset = 0;
    while let Some(start_relative) = wikitext[offset..].find("{|") {
        let start = offset + start_relative;
        let Some(end_relative) = wikitext[start..].find("\n|}") else {
            break;
        };
        let end = start + end_relative + 3;
        tables.push(&wikitext[start..end]);
        offset = end;
    }
    tables
}

fn parse_table_headers(table: &str) -> Vec<String> {
    table
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix('!'))
        .flat_map(|line| split_table_line(line, "!!"))
        .map(|cell| clean_wikitext_cell(&cell))
        .filter(|cell| !cell.is_empty())
        .collect()
}

fn parse_table_rows(table: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut current = Vec::new();
    for line in table.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("|-") {
            if !current.is_empty() {
                rows.push(current);
                current = Vec::new();
            }
        } else if trimmed.starts_with('|') && !trimmed.starts_with("|}") {
            let content = trimmed.trim_start_matches('|');
            let cells = split_table_line(content, "||");
            if cells.len() > 1 {
                current.extend(cells);
            } else if let Some(cell) = current.last_mut() {
                if !content.trim().is_empty() {
                    cell.push('\n');
                    cell.push_str(content);
                }
            } else {
                current.push(content.to_string());
            }
        } else if !current.is_empty() && !trimmed.starts_with('!') && !trimmed.starts_with("{|") {
            let Some(cell) = current.last_mut() else {
                continue;
            };
            cell.push('\n');
            cell.push_str(line);
        }
    }
    if !current.is_empty() {
        rows.push(current);
    }
    rows
}

fn split_table_line(line: &str, separator: &str) -> Vec<String> {
    line.split(separator)
        .map(strip_table_cell_attrs)
        .map(str::trim)
        .filter(|cell| !cell.is_empty())
        .map(str::to_string)
        .collect()
}

fn strip_table_cell_attrs(cell: &str) -> &str {
    let trimmed = cell.trim();
    if trimmed.contains("=\"") || trimmed.contains("width=") || trimmed.contains("style=") {
        trimmed.rsplit_once('|').map_or(trimmed, |(_, value)| value)
    } else {
        trimmed
    }
}

fn pie_column_code(header: &str) -> Option<&'static str> {
    header.eq_ignore_ascii_case("pie").then_some("ine-pro")
}

fn descendant_column(header: &str) -> Option<(&'static str, &'static str)> {
    let normalized = header
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    match normalized.trim() {
        "english" => Some(("en", "germanic")),
        "gothic" => Some(("got", "germanic")),
        "latin" => Some(("la", "italic")),
        "ancient greek" | "greek" => Some(("grc", "hellenic")),
        "sanskrit" => Some(("sa", "indo-aryan")),
        "iranian" => Some(("ira", "iranian")),
        "slavic" => Some(("sla", "slavic")),
        "baltic" => Some(("bat", "baltic")),
        "celtic" => Some(("cel", "celtic")),
        "armenian" => Some(("hy", "armenian")),
        "albanian" => Some(("sq", "albanian")),
        "tocharian" => Some(("txh", "tocharian")),
        "hittite" => Some(("hit", "anatolian")),
        _ => None,
    }
}

fn pie_descendant_language_codes() -> Vec<&'static str> {
    vec![
        "ine-pro",
        "germanic",
        "gem-pro",
        "gmw-pro",
        "gmq-pro",
        "got",
        "en",
        "enm",
        "ang",
        "sco",
        "de",
        "nl",
        "odt",
        "osx",
        "non",
        "is",
        "da",
        "sv",
        "no",
        "nb",
        "nn",
        "fy",
        "nds",
        "italic",
        "itc-pro",
        "la",
        "xum",
        "osc",
        "it",
        "fr",
        "pro",
        "es",
        "pt",
        "ro",
        "hellenic",
        "grc",
        "el",
        "indo-aryan",
        "inc-pro",
        "sa",
        "pi",
        "hi",
        "ur",
        "bn",
        "pa",
        "mr",
        "ne",
        "iranian",
        "ira-pro",
        "ae",
        "peo",
        "fa",
        "ku",
        "ps",
        "os",
        "slavic",
        "sla-pro",
        "cu",
        "ru",
        "uk",
        "pl",
        "cs",
        "sk",
        "bg",
        "sh",
        "sl",
        "baltic",
        "ine-bsl-pro",
        "bat-pro",
        "lt",
        "lv",
        "prg",
        "celtic",
        "cel-pro",
        "cel-bry-pro",
        "cel-gau",
        "sga",
        "mga",
        "ga",
        "cy",
        "wlm",
        "owl",
        "br",
        "kw",
        "armenian",
        "hy",
        "xcl",
        "albanian",
        "sqj-pro",
        "sq",
        "tocharian",
        "txh",
        "txb",
        "xto",
        "anatolian",
        "hit",
        "luw",
        "xlu",
        "xlc",
        "lyd",
        "xld",
    ]
}

fn clean_wikitext_cell(cell: &str) -> String {
    let mut text = cell.to_string();
    text = remove_between(&text, "<!--", "-->");
    text = remove_refs(&text);
    text = replace_lang_templates(&text);
    text = replace_note_templates(&text);
    text = replace_links(&text);
    text = strip_markup(&text);
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("; ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|ch: char| matches!(ch, ';' | ',' | '|'))
        .trim()
        .to_string()
}

fn remove_between(text: &str, start_marker: &str, end_marker: &str) -> String {
    let mut out = text.to_string();
    while let Some(start) = out.find(start_marker) {
        let Some(end) = out[start + start_marker.len()..]
            .find(end_marker)
            .map(|end| start + start_marker.len() + end + end_marker.len())
        else {
            out.truncate(start);
            break;
        };
        out.replace_range(start..end, "");
    }
    out
}

fn remove_refs(text: &str) -> String {
    let mut out = remove_between(text, "<ref", "</ref>");
    while let Some(start) = out.find("<ref") {
        let Some(end) = out[start..].find("/>").map(|end| start + end + 2) else {
            break;
        };
        out.replace_range(start..end, "");
    }
    out
}

fn replace_lang_templates(text: &str) -> String {
    replace_templates_by(text, |parts| {
        let name = parts.first()?.trim().to_ascii_lowercase();
        if name == "lang" || name == "langx" {
            parts.last().map(|part| part.trim().to_string())
        } else {
            None
        }
    })
}

fn replace_note_templates(text: &str) -> String {
    replace_templates_by(text, |parts| {
        let name = parts.first()?.trim().to_ascii_lowercase();
        if matches!(
            name.as_str(),
            "efn" | "refn" | "notetag" | "sfn" | "sfnp" | "cite book" | "cite journal"
        ) {
            Some(String::new())
        } else {
            None
        }
    })
}

fn replace_templates_by<F>(text: &str, mut replacement: F) -> String
where
    F: FnMut(&[String]) -> Option<String>,
{
    let mut out = String::new();
    let mut offset = 0;
    while let Some(relative_start) = text[offset..].find("{{") {
        let start = offset + relative_start;
        out.push_str(&text[offset..start]);
        let Some(end) = find_template_end(text, start) else {
            out.push_str(&text[start..]);
            return out;
        };
        let template = &text[start + 2..end];
        let parts = split_template_params(template);
        if let Some(value) = replacement(&parts) {
            out.push_str(&value);
        } else {
            out.push_str(&text[start..end + 2]);
        }
        offset = end + 2;
    }
    out.push_str(&text[offset..]);
    out
}

fn find_template_end(text: &str, start: usize) -> Option<usize> {
    let mut index = start;
    let mut depth = 0_i32;
    let bytes = text.as_bytes();
    while index + 1 < text.len() {
        match &bytes[index..index + 2] {
            b"{{" => {
                depth += 1;
                index += 2;
            }
            b"}}" => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
                index += 2;
            }
            _ => index += 1,
        }
    }
    None
}

fn replace_links(text: &str) -> String {
    let mut out = String::new();
    let mut offset = 0;
    while let Some(relative_start) = text[offset..].find("[[") {
        let start = offset + relative_start;
        out.push_str(&text[offset..start]);
        let Some(end) = text[start + 2..].find("]]").map(|end| start + 2 + end) else {
            out.push_str(&text[start..]);
            return out;
        };
        let link = &text[start + 2..end];
        let label = link.rsplit_once('|').map_or(link, |(_, label)| label);
        out.push_str(label);
        offset = end + 2;
    }
    out.push_str(&text[offset..]);
    out
}

fn strip_markup(text: &str) -> String {
    let mut out = text
        .replace("'''", "")
        .replace("''", "")
        .replace("<br />", "\n")
        .replace("<br/>", "\n")
        .replace("<br>", "\n")
        .replace("&nbsp;", " ");
    out = remove_between(&out, "<", ">");
    decode_xml_entities(&out)
}

fn extract_quoted_gloss(text: &str) -> Option<String> {
    let start = text.find('"')? + 1;
    let end = text[start..].find('"')? + start;
    let gloss = clean_wikitext_cell(&text[start..end]);
    (!gloss.is_empty()).then_some(gloss)
}

fn allowed_wiktionary_langs(config: &WiktionaryConfig) -> BTreeSet<&str> {
    config
        .languages
        .iter()
        .filter_map(|lang| wiktionary_lang_from_iso3(lang))
        .collect()
}

fn wiktionary_lang_from_iso3(lang: &str) -> Option<&'static str> {
    match lang {
        "eng" => Some("en"),
        "fra" => Some("fr"),
        "deu" => Some("de"),
        "spa" => Some("es"),
        "lat" => Some("la"),
        "ell" => Some("el"),
        "grc" => Some("grc"),
        "san" => Some("sa"),
        _ => None,
    }
}

fn iso3_from_wiktionary_lang(lang: &str) -> Option<&'static str> {
    match lang {
        "en" => Some("eng"),
        "fr" => Some("fra"),
        "de" => Some("deu"),
        "es" => Some("spa"),
        "la" => Some("lat"),
        "el" => Some("ell"),
        "grc" => Some("grc"),
        "sa" => Some("san"),
        _ => None,
    }
}

fn ipa_notation(value: &str) -> Option<&'static str> {
    if value.starts_with('/') && value.ends_with('/') {
        Some("phonemic")
    } else if value.starts_with('[') && value.ends_with(']') {
        Some("phonetic")
    } else {
        None
    }
}

fn template_named_param(params: &[String], name: &str) -> Option<String> {
    let prefix = format!("{name}=");
    params
        .iter()
        .find_map(|param| param.trim().strip_prefix(&prefix).map(str::trim))
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn find_named_templates<'a>(wikitext: &'a str, names: &[&str]) -> Vec<&'a str> {
    let mut templates = Vec::new();
    let mut offset = 0;
    while let Some(relative_start) = wikitext[offset..].find("{{") {
        let start = offset + relative_start + 2;
        let Some(name_end) = wikitext[start..].find('|').map(|end| start + end) else {
            offset = start;
            continue;
        };
        let found_name = &wikitext[start..name_end];
        if !names
            .iter()
            .any(|name| found_name.eq_ignore_ascii_case(name))
        {
            offset = start;
            continue;
        }
        let mut index = start;
        let mut depth = 1_i32;
        let bytes = wikitext.as_bytes();
        while index + 1 < wikitext.len() {
            match &bytes[index..index + 2] {
                b"{{" => {
                    depth += 1;
                    index += 2;
                }
                b"}}" => {
                    depth -= 1;
                    if depth == 0 {
                        templates.push(&wikitext[start..index]);
                        offset = index + 2;
                        break;
                    }
                    index += 2;
                }
                _ => index += 1,
            }
        }
        if depth != 0 {
            break;
        }
    }
    templates
}

fn split_template_params(template: &str) -> Vec<String> {
    let mut params = Vec::new();
    let mut current = String::new();
    let mut curly_depth = 0_i32;
    let mut link_depth = 0_i32;
    let chars = template.chars().collect::<Vec<_>>();
    let mut index = 0;
    while index < chars.len() {
        if index + 1 < chars.len() && chars[index] == '{' && chars[index + 1] == '{' {
            curly_depth += 1;
            current.push(chars[index]);
            current.push(chars[index + 1]);
            index += 2;
        } else if index + 1 < chars.len() && chars[index] == '}' && chars[index + 1] == '}' {
            curly_depth -= 1;
            current.push(chars[index]);
            current.push(chars[index + 1]);
            index += 2;
        } else if index + 1 < chars.len() && chars[index] == '[' && chars[index + 1] == '[' {
            link_depth += 1;
            current.push(chars[index]);
            current.push(chars[index + 1]);
            index += 2;
        } else if index + 1 < chars.len() && chars[index] == ']' && chars[index + 1] == ']' {
            link_depth -= 1;
            current.push(chars[index]);
            current.push(chars[index + 1]);
            index += 2;
        } else if chars[index] == '|' && curly_depth == 0 && link_depth == 0 {
            params.push(current.trim().to_string());
            current.clear();
            index += 1;
        } else {
            current.push(chars[index]);
            index += 1;
        }
    }
    params.push(current.trim().to_string());
    params
}

fn decode_xml_entities(value: &str) -> String {
    let mut decoded = value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&");
    while let Some(start) = decoded.find("&#") {
        let Some(end) = decoded[start..].find(';').map(|end| start + end) else {
            break;
        };
        let entity = &decoded[start + 2..end];
        let codepoint = if let Some(hex) = entity
            .strip_prefix('x')
            .or_else(|| entity.strip_prefix('X'))
        {
            u32::from_str_radix(hex, 16).ok()
        } else {
            entity.parse::<u32>().ok()
        };
        let Some(character) = codepoint.and_then(char::from_u32) else {
            break;
        };
        decoded.replace_range(start..=end, &character.to_string());
    }
    decoded
}

pub fn expand_training_examples(
    entries: &[PronunciationEntry],
    config: &WiktionaryConfig,
) -> Vec<TrainingExample> {
    let mut examples = Vec::new();
    expand_training_examples_to(entries, config, &mut |_| {}, None, |example| {
        examples.push(example);
        Ok(())
    })
    .expect("collecting expanded training examples should not fail");
    examples
}

fn expand_training_examples_to(
    entries: &[PronunciationEntry],
    config: &WiktionaryConfig,
    progress: &mut impl FnMut(PrepareProgress),
    progress_path: Option<&Path>,
    mut emit: impl FnMut(TrainingExample) -> Result<()>,
) -> Result<()> {
    let allowed: BTreeSet<&str> = config.languages.iter().map(String::as_str).collect();
    let mut seen_normalize = HashSet::new();
    let normalized_entries = entries
        .iter()
        .filter(|entry| allowed.contains(entry.lang.as_str()))
        .map(NormalizedPronunciationEntry::from)
        .collect::<Vec<_>>();
    let mut emitted = 0_usize;

    for (index, row) in normalized_entries.iter().enumerate() {
        let entry = row.entry;
        let controls = wiktionary_training_controls(
            WiktionaryTask::OrthographyToPhonology,
            &entry.lang,
            Some(row.representation),
            row.variety.as_deref(),
        );
        let source = pronunciation_entry_source(entry);
        emit(TrainingExample {
            task: WiktionaryTask::OrthographyToPhonology,
            lang: Some(entry.lang.clone()),
            notation: Some(entry.notation.clone()),
            accent: row.variety.clone(),
            input: format!("{controls} {}", row.orthography),
            output: row.pronunciation.clone(),
            source: source.clone(),
        })?;
        emitted += 1;
        if config.include_reverse {
            let controls = wiktionary_training_controls(
                WiktionaryTask::PhonologyToOrthography,
                &entry.lang,
                Some(row.representation),
                row.variety.as_deref(),
            );
            emit(TrainingExample {
                task: WiktionaryTask::PhonologyToOrthography,
                lang: Some(entry.lang.clone()),
                notation: Some(entry.notation.clone()),
                accent: row.variety.clone(),
                input: format!("{controls} {}", row.pronunciation),
                output: row.orthography.clone(),
                source: source.clone(),
            })?;
            emitted += 1;
        }
        if seen_normalize.insert(format!("{}\t{}", entry.lang, row.orthography)) {
            emit(TrainingExample {
                task: WiktionaryTask::NormalizeText,
                lang: Some(entry.lang.clone()),
                notation: None,
                accent: None,
                input: format!(
                    "{} <lang:{}> {}",
                    WiktionaryTask::NormalizeText.token(),
                    entry.lang,
                    row.orthography
                ),
                output: normalize_spelling_for_training(&row.orthography),
                source: source.clone(),
            })?;
            emitted += 1;
        }
        if config.include_language_guessing {
            emit(TrainingExample {
                task: WiktionaryTask::GuessLangFromOrthography,
                lang: None,
                notation: Some(entry.notation.clone()),
                accent: row.variety.clone(),
                input: format!(
                    "{} {} {}",
                    WiktionaryTask::GuessLangFromOrthography.token(),
                    row.representation,
                    row.orthography
                ),
                output: entry.lang.clone(),
                source: source.clone(),
            })?;
            emitted += 1;
            emit(TrainingExample {
                task: WiktionaryTask::GuessLangFromPhonology,
                lang: None,
                notation: Some(entry.notation.clone()),
                accent: row.variety.clone(),
                input: format!(
                    "{} {} {}",
                    WiktionaryTask::GuessLangFromPhonology.token(),
                    row.representation,
                    row.pronunciation
                ),
                output: entry.lang.clone(),
                source: source.clone(),
            })?;
            emitted += 1;
            emit(TrainingExample {
                task: WiktionaryTask::GuessLangFromOrthographyAndPhonology,
                lang: None,
                notation: Some(entry.notation.clone()),
                accent: row.variety.clone(),
                input: format!(
                    "{} {} {} => {}",
                    WiktionaryTask::GuessLangFromOrthographyAndPhonology.token(),
                    row.representation,
                    row.orthography,
                    row.pronunciation
                ),
                output: entry.lang.clone(),
                source: source.clone(),
            })?;
            emitted += 1;
        }
        maybe_report_expand_progress(progress, index + 1, emitted, progress_path);
    }

    let mut seen_realization = HashSet::new();
    for (index, phonemes) in normalized_entries
        .iter()
        .filter(|entry| entry.representation == "<repr:phonemes>")
        .enumerate()
    {
        for phones in normalized_entries.iter().filter(|entry| {
            entry.representation == "<repr:phones>"
                && entry.entry.lang == phonemes.entry.lang
                && entry.orthography == phonemes.orthography
        }) {
            let Some(variety) = compatible_realization_variety(
                phonemes.variety.as_deref(),
                phones.variety.as_deref(),
            ) else {
                continue;
            };
            let key = format!(
                "{}\t{}\t{}\t{}\t{}",
                phonemes.entry.lang,
                phonemes.orthography,
                variety.unwrap_or(""),
                phonemes.pronunciation,
                phones.pronunciation
            );
            if !seen_realization.insert(key) {
                continue;
            }
            let controls = wiktionary_training_controls(
                WiktionaryTask::PhoneticRealization,
                &phonemes.entry.lang,
                Some("<repr:phonemes>"),
                variety,
            );
            emit(TrainingExample {
                task: WiktionaryTask::PhoneticRealization,
                lang: Some(phonemes.entry.lang.clone()),
                notation: Some("phonetic-realization".to_string()),
                accent: variety.map(str::to_string),
                input: format!("{controls} {}", phonemes.pronunciation),
                output: phones.pronunciation.clone(),
                source: format!(
                    "{}+{}",
                    pronunciation_entry_source(phonemes.entry),
                    pronunciation_entry_source(phones.entry)
                ),
            })?;
            emitted += 1;
        }
        maybe_report_expand_progress(
            progress,
            normalized_entries.len() + index + 1,
            emitted,
            progress_path,
        );
    }

    progress(PrepareProgress::Expand {
        rows: entries.len(),
        examples: emitted,
        path: progress_path.map(|path| path.display().to_string()),
    });
    Ok(())
}

fn maybe_report_expand_progress(
    progress: &mut impl FnMut(PrepareProgress),
    rows: usize,
    examples: usize,
    path: Option<&Path>,
) {
    if rows <= 10 || rows % 10_000 == 0 {
        progress(PrepareProgress::Expand {
            rows,
            examples,
            path: path.map(|path| path.display().to_string()),
        });
    }
}

struct NormalizedPronunciationEntry<'a> {
    entry: &'a PronunciationEntry,
    orthography: String,
    pronunciation: String,
    representation: &'static str,
    variety: Option<String>,
}

impl<'a> From<&'a PronunciationEntry> for NormalizedPronunciationEntry<'a> {
    fn from(entry: &'a PronunciationEntry) -> Self {
        Self {
            entry,
            orthography: normalize_orthography_for_training(&entry.spelling),
            pronunciation: normalize_ipa_for_training(&entry.ipa),
            representation: wiktionary_representation_token(&entry.notation),
            variety: entry
                .accent
                .as_deref()
                .map(|accent| canonicalize_accent(&entry.lang, accent))
                .filter(|accent| !accent.is_empty()),
        }
    }
}

fn compatible_realization_variety<'a>(
    phonemes: Option<&'a str>,
    phones: Option<&'a str>,
) -> Option<Option<&'a str>> {
    match (phonemes, phones) {
        (Some(phonemes), Some(phones)) if phonemes == phones => Some(Some(phones)),
        (Some(_), Some(_)) => None,
        (Some(phonemes), None) => Some(Some(phonemes)),
        (None, Some(phones)) => Some(Some(phones)),
        (None, None) => Some(None),
    }
}

fn pronunciation_entry_source(entry: &PronunciationEntry) -> String {
    if entry.raw_template.starts_with("{{synthetic-spanish|") {
        "synthetic-spanish-orthography+enwiktionary-title".to_string()
    } else if entry.raw_template.starts_with("{{wiktionary-supplement|") {
        "wiktionary-supplement".to_string()
    } else {
        "enwiktionary".to_string()
    }
}

pub fn expand_pie_training_examples(
    entries: &[PieEtymologyEntry],
    config: &WiktionaryConfig,
) -> Vec<TrainingExample> {
    let allowed: BTreeSet<&str> = config.languages.iter().map(String::as_str).collect();
    let mut examples = Vec::new();
    let eligible = entries
        .iter()
        .filter(|entry| {
            allowed.is_empty()
                || allowed.contains(entry.lang.as_str())
                || allowed.contains(entry.branch.as_str())
        })
        .collect::<Vec<_>>();

    for entry in &eligible {
        if !allowed.is_empty()
            && !allowed.contains(entry.lang.as_str())
            && !allowed.contains(entry.branch.as_str())
        {
            continue;
        }
        examples.push(TrainingExample {
            task: WiktionaryTask::EtymologyTranslation,
            lang: Some(entry.lang.clone()),
            notation: Some("etymology".to_string()),
            accent: None,
            input: etymology_translation_input("ine-pro", &entry.lang, &entry.pie),
            output: entry.descendant.clone(),
            source: entry.source.clone(),
        });

        if config.include_reverse {
            examples.push(TrainingExample {
                task: WiktionaryTask::EtymologyTranslation,
                lang: Some("ine-pro".to_string()),
                notation: Some("etymology".to_string()),
                accent: None,
                input: etymology_translation_input(&entry.lang, "ine-pro", &entry.descendant),
                output: entry.pie.clone(),
                source: entry.source.clone(),
            });
        }
    }

    if config.include_descendant_pairs {
        let mut seen = HashSet::new();
        for source in &eligible {
            for target in &eligible {
                if source.pie != target.pie
                    || source.lang == target.lang && source.descendant == target.descendant
                {
                    continue;
                }
                let key = format!(
                    "{}\t{}\t{}\t{}\t{}",
                    source.pie, source.lang, source.descendant, target.lang, target.descendant
                );
                if !seen.insert(key) {
                    continue;
                }
                examples.push(TrainingExample {
                    task: WiktionaryTask::EtymologyTranslation,
                    lang: Some(target.lang.clone()),
                    notation: Some("etymology".to_string()),
                    accent: None,
                    input: etymology_translation_input(
                        &source.lang,
                        &target.lang,
                        &source.descendant,
                    ),
                    output: target.descendant.clone(),
                    source: format!("{}+{}", source.source, target.source),
                });
            }
        }
    }
    examples
}

fn etymology_translation_input(source_lang: &str, target_lang: &str, word: &str) -> String {
    format!(
        "{} <from:{source_lang}> <to:{target_lang}> {word}",
        WiktionaryTask::EtymologyTranslation.token()
    )
}

pub fn normalize_ipa_for_training(ipa: &str) -> String {
    let trimmed = ipa.trim();
    let payload = if (trimmed.starts_with('/') && trimmed.ends_with('/'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        trimmed[1..trimmed.len() - 1].trim()
    } else {
        trimmed
    };
    payload.nfc().collect()
}

pub fn normalize_orthography_for_training(orthography: &str) -> String {
    orthography.trim().nfc().collect()
}

pub fn canonicalize_accent(lang: &str, accent: &str) -> String {
    let trimmed = accent.trim();
    if lang == "eng" {
        match trimmed {
            "GA" | "GenAm" => "en-US.GenAm".to_string(),
            "RP" => "en-GB.RP".to_string(),
            "SSB" => "en-GB.SSB".to_string(),
            "IE" => "en-IE".to_string(),
            "Dublin / East" => "en-IE.Dublin.East".to_string(),
            "Local Dublin" => "en-IE.Dublin.Local".to_string(),
            _ => canonical_tag_fragment(trimmed),
        }
    } else {
        canonical_tag_fragment(trimmed)
    }
}

pub fn canonicalize_training_tag_value(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || matches!(ch, '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

pub fn normalize_spelling_for_training(spelling: &str) -> String {
    normalize_orthography_for_training(spelling)
        .to_lowercase()
        .nfc()
        .collect()
}

fn canonical_tag_fragment(value: &str) -> String {
    value
        .split(|c: char| c.is_whitespace() || matches!(c, '/' | ',' | ';' | '|'))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(".")
}

fn wiktionary_training_controls(
    task: WiktionaryTask,
    lang: &str,
    representation: Option<&str>,
    variety: Option<&str>,
) -> String {
    let mut controls = format!("{} <lang:{lang}>", task.token());
    if let Some(variety) = variety.filter(|variety| !variety.is_empty()) {
        controls.push(' ');
        controls.push_str(&format!("<variety:{variety}>"));
    }
    if let Some(representation) = representation.filter(|representation| !representation.is_empty())
    {
        controls.push(' ');
        controls.push_str(wiktionary_representation_token(representation));
    }
    controls
}

pub fn wiktionary_representation_token(notation: &str) -> &'static str {
    match notation {
        "<repr:phonemes>" | "phonemic" | "phoneme" | "phonemes" => "<repr:phonemes>",
        "<repr:phones>" | "phonetic" | "phone" | "phones" => "<repr:phones>",
        "<repr:diaphonemes>" | "diaphonemic" | "diaphoneme" | "diaphonemes" => "<repr:diaphonemes>",
        _ => "<repr:unknown>",
    }
}

pub fn normalize_wiktionary_control_tokens(input: &str) -> String {
    input.to_string()
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

fn write_jsonl_with_progress<T: Serialize>(
    path: &Path,
    examples: &[T],
    progress: &mut impl FnMut(PrepareProgress),
) -> Result<()> {
    progress(PrepareProgress::Stage {
        message: format!("Writing {} rows to {}", examples.len(), path.display()),
    });
    let mut file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    for example in examples {
        writeln!(file, "{}", serde_json::to_string(example)?)?;
    }
    progress(PrepareProgress::Write {
        path: path.display().to_string(),
        rows: examples.len(),
    });
    Ok(())
}

fn write_vocab(out: &Path, examples: &[TrainingExample]) -> Result<()> {
    let inputs = examples
        .iter()
        .map(|example| example.input.clone())
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
        "# Wiktionary pronunciation dataset\n\nSource dump: `{}`\n\nConfigured languages: {}\n\n`phonemes.jsonl` contains slash-delimited phonemic `{{IPA|...|/.../}}` rows. `phones.jsonl` contains bracket-delimited phonetic `{{IPA|...|[...]}}` rows. Both preserve raw orthography, IPA text, notation, accent/variety metadata, and the raw template. `patterns.jsonl` keeps other useful pronunciation-section templates such as audio, homophones, and rhymes. `train.jsonl`, `valid.jsonl`, and `test.jsonl` expand those rows into NFC-normalized model-facing tasks.\n\nTraining row shapes:\n\n```text\n<task:orthography_to_phonology> <lang:eng> <repr:phonemes> disease => dəˈziːz\n<task:orthography_to_phonology> <lang:eng> <variety:en-GB.RP> <repr:phones> Ireland => ˈɑɪələnd\n<task:orthography_to_phonology> <lang:deu> <repr:phones> Honduras => hɔnˈduːʁas\n<task:phonology_to_orthography> <lang:eng> <repr:phonemes> dəˈziːz => disease\n<task:phonetic_realization> <lang:eng> <variety:en-GB.RP> <repr:phonemes> ˈaɪələnd => ˈɑɪələnd\n<task:align> <lang:eng> audio_features + text => phone_times\n<task:normalize> <lang:eng> Disease! => disease\n```\n\nRepresentation tokens preserve the phonemes/phones distinction while targets omit only the outer visual delimiters. Variety tags are compact token-safe labels such as `en-US.GenAm`, `en-GB.RP`, and `Castilian`. Phonetic-realization rows are emitted only when matched phonemic and phonetic source rows exist for the same normalized orthography, language, and compatible variety metadata. Reverse and language-guessing rows are controlled by `include_reverse` and `include_language_guessing`; align rows require audio timing data and are reserved for datasets that provide it.\n",
        dump_path.display(),
        config.languages.join(", ")
    )
}

fn pie_dataset_readme(config: &WiktionaryConfig, source_paths: &[PathBuf]) -> String {
    format!(
        "# Wiktionary PIE etymology dataset\n\nSource pages: `{}`\n\nConfigured languages: {}\n\n`pie_roots.jsonl` contains reconstructed Proto-Indo-European roots or words paired with descendant and cognate forms from Wiktionary etymology/root/descendant templates, plus any configured supplemental Wikipedia tables. `train.jsonl`, `valid.jsonl`, and `test.jsonl` expand those pairs into one model-facing translation task:\n\n```text\n<task:etymology_translate> <from:ine-pro> <to:la> *meh2ter => mater\n<task:etymology_translate> <from:la> <to:ine-pro> mater => *meh2ter\n<task:etymology_translate> <from:en> <to:de> thorp => Dorf\n```\n\nThe configured language list includes PIE (`ine-pro`) plus major Indo-European branches, proto-languages, historical witnesses, and common modern descendants using Wiktionary language codes.\n",
        source_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("`, `"),
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
        assert_eq!(
            config.languages,
            ["eng", "fra", "deu", "spa", "lat", "ell", "grc", "san"]
        );
        assert_eq!(config.train_task, "all");
        assert_eq!(config.train_notations, ["phonemic", "phonetic"]);
    }

    #[test]
    fn expands_orthography_phonology_and_language_guessing_tasks() {
        let config = WiktionaryConfig::default();
        let examples = expand_training_examples(
            &[PronunciationEntry {
                lang: "deu".to_string(),
                wiktionary_lang: "de".to_string(),
                spelling: "schief".to_string(),
                ipa: "/ʃiːf/".to_string(),
                notation: "phonemic".to_string(),
                accent: None,
                raw_template: "{{IPA|de|/ʃiːf/}}".to_string(),
            }],
            &config,
        );
        assert_eq!(examples.len(), 6);
        assert!(examples.iter().any(|example| {
            example.task == WiktionaryTask::OrthographyToPhonology
                && example.input
                    == "<task:orthography_to_phonology> <lang:deu> <repr:phonemes> schief"
                && example.output == "ʃiːf"
        }));
        assert!(examples.iter().any(|example| {
            example.task == WiktionaryTask::PhonologyToOrthography
                && example.input
                    == "<task:phonology_to_orthography> <lang:deu> <repr:phonemes> ʃiːf"
                && example.output == "schief"
        }));
        assert!(examples.iter().any(|example| {
            example.task == WiktionaryTask::NormalizeText
                && example.input == "<task:normalize> <lang:deu> schief"
                && example.output == "schief"
        }));
    }

    #[test]
    fn extracts_pie_root_templates_from_wiktionary_descendant_pages() {
        let config = WiktionaryConfig::pie_etymology();
        let entries = extract_wiktionary_pie_etymology_entries(
            "thorp",
            r#"
==English==

===Etymology===
{{root|en|ine-pro|*treb-}}
From {{inh|en|enm|thorp}}, from {{inh|en|ang|þorp}}, from {{der|en|ine-pro|*trab-}}, {{m|ine-pro|*treb-|t=dwelling, room}}.

===Noun===
# A hamlet.
"#,
            &config,
        );

        assert!(entries.iter().any(|entry| {
            entry.pie == "*treb-" && entry.lang == "en" && entry.descendant == "thorp"
        }));
        assert!(entries.iter().any(|entry| {
            entry.pie == "*trab-" && entry.lang == "en" && entry.descendant == "thorp"
        }));
    }

    #[test]
    fn extracts_pie_reconstruction_descendants() {
        let config = WiktionaryConfig::pie_etymology();
        let entries = extract_wiktionary_pie_etymology_entries(
            "Reconstruction:Proto-Indo-European/treb-",
            r#"
{{reconstructed}}
==Proto-Indo-European==
{{etymon|ine-pro|pos=root}}

===Root===
{{ine-root}}

# [[settlement]], [[dwelling]]

====Derived terms====
* {{l|ine-pro||*treb-eh₂}}
** {{desc|cel-pro|*trebā|t=settlement}} {{see desc}}
* {{l|ine-pro||*tr̥b-om}}
** {{desc|gem-pro|*þurpą}} {{see desc}}
"#,
            &config,
        );

        assert!(entries.iter().any(|entry| {
            entry.pie == "*treb-" && entry.lang == "cel-pro" && entry.descendant == "*trebā"
        }));
        assert!(entries.iter().any(|entry| {
            entry.pie == "*treb-" && entry.lang == "gem-pro" && entry.descendant == "*þurpą"
        }));
    }

    #[test]
    fn ignores_weak_pie_mentions_and_meta_pages() {
        let config = WiktionaryConfig::pie_etymology();
        let entries = extract_wiktionary_pie_etymology_entries(
            "thing",
            r#"
==English==
{{bor|en|ine-pro|*bʰer-}}
{{cog|ine-pro|*bʰer-}}
{{m|ine-pro|*bʰer-|t=carry}}
{{l|ine-pro|*bʰer-}}
{{inh|en|ine-pro|*dʰeh₁-}}
"#,
            &config,
        );

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].pie, "*dʰeh₁-");
        assert_eq!(entries[0].descendant, "thing");

        let meta_entries = extract_wiktionary_pie_etymology_entries(
            "Category:Icelandic terms inherited from PIE root",
            "==Icelandic==\n{{inh|is|ine-pro|*h₁es-}}",
            &config,
        );
        assert!(meta_entries.is_empty());
    }

    #[test]
    fn rejects_placeholder_pie_descendants() {
        let config = WiktionaryConfig::pie_etymology();
        let entries = extract_wiktionary_pie_etymology_entries(
            "Reconstruction:Proto-Indo-European/h₁es-",
            r#"
==Proto-Indo-European==
{{etymon|ine-pro|pos=root}}
* {{desc|gem-pro|-}}
* {{desc|grc|inherited from PIE root}}
* {{desc|la|est}}
"#,
            &config,
        );

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].descendant, "est");
    }

    #[test]
    fn expands_pie_pairs_in_both_directions() {
        let mut config = WiktionaryConfig::pie_etymology();
        config.include_descendant_pairs = true;
        let examples = expand_pie_training_examples(
            &[
                PieEtymologyEntry {
                    pie: "*treb-".to_string(),
                    lang: "en".to_string(),
                    branch: "germanic".to_string(),
                    descendant: "thorp".to_string(),
                    gloss: Some("dwelling, room".to_string()),
                    source: "test".to_string(),
                },
                PieEtymologyEntry {
                    pie: "*treb-".to_string(),
                    lang: "de".to_string(),
                    branch: "germanic".to_string(),
                    descendant: "Dorf".to_string(),
                    gloss: Some("village".to_string()),
                    source: "test".to_string(),
                },
            ],
            &config,
        );

        assert!(examples.iter().any(|example| {
            example.task == WiktionaryTask::EtymologyTranslation
                && example.input == "<task:etymology_translate> <from:ine-pro> <to:en> *treb-"
                && example.output == "thorp"
        }));
        assert!(examples.iter().any(|example| {
            example.task == WiktionaryTask::EtymologyTranslation
                && example.input == "<task:etymology_translate> <from:en> <to:ine-pro> thorp"
                && example.output == "*treb-"
        }));
        assert!(examples.iter().any(|example| {
            example.task == WiktionaryTask::EtymologyTranslation
                && example.input == "<task:etymology_translate> <from:en> <to:de> thorp"
                && example.output == "Dorf"
        }));
    }

    #[test]
    fn pie_config_does_not_expand_descendant_pairs_by_default() {
        let config = WiktionaryConfig::pie_etymology();
        let examples = expand_pie_training_examples(
            &[
                PieEtymologyEntry {
                    pie: "*treb-".to_string(),
                    lang: "en".to_string(),
                    branch: "germanic".to_string(),
                    descendant: "thorp".to_string(),
                    gloss: None,
                    source: "test".to_string(),
                },
                PieEtymologyEntry {
                    pie: "*treb-".to_string(),
                    lang: "de".to_string(),
                    branch: "germanic".to_string(),
                    descendant: "Dorf".to_string(),
                    gloss: None,
                    source: "test".to_string(),
                },
            ],
            &config,
        );

        assert!(!examples.iter().any(|example| {
            example.input == "<task:etymology_translate> <from:en> <to:de> thorp"
        }));
    }

    #[test]
    fn normalizes_pronunciation_payloads_and_orthography_with_nfc() {
        assert_eq!(normalize_ipa_for_training("/e\u{301}/"), "é");
        assert_eq!(normalize_ipa_for_training("[i\u{308}]"), "ï");
        assert_eq!(normalize_orthography_for_training(" cafe\u{301} "), "café");
        assert_eq!(normalize_spelling_for_training(" Cafe\u{301} "), "café");
    }

    #[test]
    fn formats_representation_variety_reverse_and_normalize_training_controls() {
        let config = WiktionaryConfig::default();
        let examples = expand_training_examples(
            &[PronunciationEntry {
                lang: "eng".to_string(),
                wiktionary_lang: "en".to_string(),
                spelling: "Ireland".to_string(),
                ipa: "[ˈäɪɚɫɪ̈nd]".to_string(),
                notation: "phonetic".to_string(),
                accent: Some("GenAm".to_string()),
                raw_template: "{{IPA|en|[ˈäɪɚɫɪ̈nd]|a=GenAm}}".to_string(),
            }],
            &config,
        );

        let forward = examples
            .iter()
            .find(|example| example.task == WiktionaryTask::OrthographyToPhonology)
            .expect("forward example");
        assert_eq!(
            forward.input,
            "<task:orthography_to_phonology> <lang:eng> <variety:en-US.GenAm> <repr:phones> Ireland"
        );
        assert_eq!(forward.output, "ˈäɪɚɫɪ̈nd");
        assert_eq!(forward.notation.as_deref(), Some("phonetic"));
        assert_eq!(forward.accent.as_deref(), Some("en-US.GenAm"));

        let reverse = examples
            .iter()
            .find(|example| example.task == WiktionaryTask::PhonologyToOrthography)
            .expect("reverse example");
        assert_eq!(
            reverse.input,
            "<task:phonology_to_orthography> <lang:eng> <variety:en-US.GenAm> <repr:phones> ˈäɪɚɫɪ̈nd"
        );
        assert_eq!(reverse.output, "Ireland");

        let normalize = examples
            .iter()
            .find(|example| example.task == WiktionaryTask::NormalizeText)
            .expect("normalize example");
        assert_eq!(normalize.input, "<task:normalize> <lang:eng> Ireland");
        assert_eq!(normalize.output, "ireland");

        assert_eq!(canonicalize_training_tag_value("weak vowel"), "weak_vowel");
        assert_eq!(
            canonicalize_training_tag_value("Dublin / East"),
            "Dublin_East"
        );
    }

    #[test]
    fn emits_phonetic_realization_for_matched_phoneme_and_phone_rows() {
        let config = WiktionaryConfig {
            include_language_guessing: false,
            ..WiktionaryConfig::default()
        };
        let examples = expand_training_examples(
            &[
                PronunciationEntry {
                    lang: "eng".to_string(),
                    wiktionary_lang: "en".to_string(),
                    spelling: "Ireland".to_string(),
                    ipa: "/ˈaɪərlənd/".to_string(),
                    notation: "phonemic".to_string(),
                    accent: Some("GenAm".to_string()),
                    raw_template: "{{IPA|en|/ˈaɪərlənd/|a=GenAm}}".to_string(),
                },
                PronunciationEntry {
                    lang: "eng".to_string(),
                    wiktionary_lang: "en".to_string(),
                    spelling: "Ireland".to_string(),
                    ipa: "[ˈäɪɚɫɪ̈nd]".to_string(),
                    notation: "phonetic".to_string(),
                    accent: Some("GenAm".to_string()),
                    raw_template: "{{IPA|en|[ˈäɪɚɫɪ̈nd]|a=GenAm}}".to_string(),
                },
            ],
            &config,
        );

        let realization = examples
            .iter()
            .find(|example| example.task == WiktionaryTask::PhoneticRealization)
            .expect("phonetic realization example");
        assert_eq!(
            realization.input,
            "<task:phonetic_realization> <lang:eng> <variety:en-US.GenAm> <repr:phonemes> ˈaɪərlənd"
        );
        assert_eq!(realization.output, "ˈäɪɚɫɪ̈nd");
        assert_eq!(realization.accent.as_deref(), Some("en-US.GenAm"));
    }

    #[test]
    fn finds_dump_href_from_index() {
        let index = r#"<a href="enwiktionary-20260601-pages-meta-current.xml.bz2">dump</a>"#;
        assert_eq!(
            find_dump_href(index),
            Some("enwiktionary-20260601-pages-meta-current.xml.bz2")
        );
    }

    #[test]
    fn extracts_ipa_audio_homophone_and_rhyme_patterns_from_page() {
        let config = WiktionaryConfig::default();
        let text = r#"==English==
===Pronunciation===
* {{enPR|frē}}, {{IPA|en|/fɹiː/|[fɹɪi̯]|a=RP}}
* {{audio|en|En-uk-free.ogg|a=RP}}
* {{IPA|en|/fɹi/|a=GA}}
* {{homophones|en|three|aa=th-fronting}}
* {{rhymes|en|iː|s=1}}
"#;

        let data = extract_page_data("free", text, &config);

        assert_eq!(data.phonemes.len(), 2);
        assert_eq!(data.phones.len(), 1);
        assert!(data.phonemes.iter().any(|entry| {
            entry.lang == "eng"
                && entry.wiktionary_lang == "en"
                && entry.spelling == "free"
                && entry.ipa == "/fɹiː/"
                && entry.notation == "phonemic"
                && entry.accent.as_deref() == Some("RP")
        }));
        assert!(data.phones.iter().any(|entry| {
            entry.lang == "eng"
                && entry.wiktionary_lang == "en"
                && entry.spelling == "free"
                && entry.ipa == "[fɹɪi̯]"
                && entry.notation == "phonetic"
                && entry.accent.as_deref() == Some("RP")
        }));
        assert!(data.patterns.iter().any(|pattern| pattern.kind == "audio"
            && pattern.values == ["En-uk-free.ogg"]
            && pattern.accent.as_deref() == Some("RP")));
        assert!(data
            .patterns
            .iter()
            .any(|pattern| pattern.kind == "homophones" && pattern.values == ["three"]));
        assert!(data
            .patterns
            .iter()
            .any(|pattern| pattern.kind == "rhymes" && pattern.values == ["iː"]));
    }

    #[test]
    fn synthesizes_spanish_pronunciations_from_page_titles() {
        let config = WiktionaryConfig {
            languages: vec!["spa".to_string()],
            include_language_guessing: false,
            ..WiktionaryConfig::default()
        };
        let text = r#"==Spanish==
===Noun===
{{es-noun|m}}
"#;

        let data = extract_page_data("zapato", text, &config);

        assert_eq!(data.phonemes.len(), 2);
        assert!(data.phonemes.iter().any(|entry| {
            entry.lang == "spa"
                && entry.accent.as_deref() == Some("Castilian")
                && entry.ipa == "/θaˈpato/"
                && entry.raw_template.starts_with("{{synthetic-spanish|")
        }));
        assert!(data.phonemes.iter().any(|entry| {
            entry.lang == "spa"
                && entry.accent.as_deref() == Some("LatAm")
                && entry.ipa == "/saˈpato/"
        }));

        let examples = expand_training_examples(&data.phonemes, &config);
        assert!(examples.iter().any(|example| {
            example.task == WiktionaryTask::OrthographyToPhonology
                && example.input
                    == "<task:orthography_to_phonology> <lang:spa> <variety:Castilian> <repr:phonemes> zapato"
                && example.output == "θaˈpato"
                && example.source == "synthetic-spanish-orthography+enwiktionary-title"
        }));
    }

    #[test]
    fn parses_bzip2_xml_dump() {
        let config = WiktionaryConfig {
            max_pages: Some(1),
            ..WiktionaryConfig::default()
        };
        let xml = r#"<mediawiki>
  <page>
    <title>free</title>
    <revision>
      <text xml:space="preserve">==English==
===Pronunciation===
* {{IPA|en|/fɹiː/|[fɹɪi̯]|a=RP}}
</text>
    </revision>
  </page>
</mediawiki>
"#;
        let path = std::env::temp_dir().join(format!(
            "tongues-wiktionary-test-{}.xml.bz2",
            std::process::id()
        ));
        let file = File::create(&path).expect("create compressed fixture");
        let mut encoder = bzip2::write::BzEncoder::new(file, bzip2::Compression::best());
        encoder.write_all(xml.as_bytes()).expect("write fixture");
        encoder.finish().expect("finish fixture");

        let data = parse_dump(&path, &config).expect("parse compressed dump");
        let _ = fs::remove_file(&path);

        assert_eq!(data.phonemes.len(), 1);
        assert_eq!(data.phones.len(), 1);
        assert_eq!(data.phonemes[0].ipa, "/fɹiː/");
        assert_eq!(data.phones[0].ipa, "[fɹɪi̯]");
    }
}
