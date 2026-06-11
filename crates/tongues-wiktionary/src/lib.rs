//! Wiktionary pronunciation model-family data preparation.
//!
//! This family downloads the English Wiktionary MediaWiki XML dump and expands
//! extracted spelling/IPA pairs into multilingual seq2seq-style training rows.
//! The XML/wikitext extraction itself is intentionally stubbed until the parser
//! policy for Wiktionary pronunciation templates is implemented.

use std::collections::{BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
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
    #[serde(default)]
    pub dump_path: Option<String>,
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
            dump_path: None,
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
    pub extracted_patterns: usize,
    pub parsed_phonemes: usize,
    pub parsed_phones: usize,
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

    let dump_path = resolve_dump_path(cache_dir, config)?;
    let extracted = parse_dump(&dump_path, config)?;
    let phonemes = extracted.phonemes;
    let phones = extracted.phones;
    let examples = expand_training_examples(&phonemes, config);
    let (train, valid, test) =
        split_examples(examples, config.train_frac, config.valid_frac, config.seed);

    write_jsonl(&out.join("train.jsonl"), &train)?;
    write_jsonl(&out.join("valid.jsonl"), &valid)?;
    write_jsonl(&out.join("test.jsonl"), &test)?;
    write_jsonl(&out.join("patterns.jsonl"), &extracted.patterns)?;
    write_jsonl(&out.join("phonemes.jsonl"), &phonemes)?;
    write_jsonl(&out.join("phones.jsonl"), &phones)?;
    write_vocab(out, [&train[..], &valid[..], &test[..]].concat().as_slice())?;
    fs::write(
        out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(out.join("README.md"), dataset_readme(config, &dump_path))?;

    Ok(PrepareReport {
        dump_path: dump_path.display().to_string(),
        extracted_patterns: extracted.patterns.len(),
        parsed_phonemes: phonemes.len(),
        parsed_phones: phones.len(),
        train_examples: train.len(),
        valid_examples: valid.len(),
        test_examples: test.len(),
    })
}

pub fn resolve_dump_path(cache_dir: &Path, config: &WiktionaryConfig) -> Result<PathBuf> {
    if let Some(path) = &config.dump_path {
        return Ok(PathBuf::from(path));
    }
    download_dump(cache_dir, config)
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

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ExtractedWiktionaryData {
    pub patterns: Vec<WiktionaryPattern>,
    pub phonemes: Vec<PronunciationEntry>,
    pub phones: Vec<PronunciationEntry>,
}

pub fn parse_dump(dump_path: &Path, config: &WiktionaryConfig) -> Result<ExtractedWiktionaryData> {
    let file = File::open(dump_path).with_context(|| format!("opening {}", dump_path.display()))?;
    if dump_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension == "bz2")
    {
        let decoder = BzDecoder::new(file);
        let reader = BufReader::with_capacity(1024 * 1024, decoder);
        parse_xml_pages(reader, config)
    } else {
        let reader = BufReader::with_capacity(1024 * 1024, file);
        parse_xml_pages(reader, config)
    }
}

fn parse_xml_pages<R: BufRead>(
    reader: R,
    config: &WiktionaryConfig,
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
            if config.max_pages.is_some_and(|max| pages_seen >= max) {
                break;
            }
        } else {
            text.push_str(&decode_xml_entities(&line));
            text.push('\n');
        }
    }

    Ok(data)
}

impl ExtractedWiktionaryData {
    fn extend(&mut self, other: ExtractedWiktionaryData) {
        self.patterns.extend(other.patterns);
        self.phonemes.extend(other.phonemes);
        self.phones.extend(other.phones);
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
    if spelling.is_empty() || spelling.contains(':') {
        return ExtractedWiktionaryData::default();
    }

    let allowed = allowed_wiktionary_langs(config);
    let mut data = ExtractedWiktionaryData::default();
    let mut seen = HashSet::new();
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
    data
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
        _ => None,
    }
}

fn iso3_from_wiktionary_lang(lang: &str) -> Option<&'static str> {
    match lang {
        "en" => Some("eng"),
        "fr" => Some("fra"),
        "de" => Some("deu"),
        "es" => Some("spa"),
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

fn write_jsonl<T: Serialize>(path: &Path, examples: &[T]) -> Result<()> {
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
        "# Wiktionary pronunciation dataset\n\nSource dump: `{}`\n\nConfigured languages: {}\n\n`phonemes.jsonl` contains slash-delimited phonemic `{{IPA|...|/.../}}` rows. `phones.jsonl` contains bracket-delimited phonetic `{{IPA|...|[...]}}` rows. Both preserve language, spelling, IPA text, notation, accent metadata, and the raw template. `patterns.jsonl` keeps other useful pronunciation-section templates such as audio, homophones, and rhymes. `train.jsonl`, `valid.jsonl`, and `test.jsonl` currently expand phoneme rows into spelling-to-IPA, IPA-to-spelling, and optional language-guessing tasks.\n\nTraining row shape:\n\n```json\n{{\"task\":\"spelling-to-ipa\",\"lang\":\"eng\",\"input\":\"lang=eng | champ\",\"output\":\"/ʃɑ̃/\",\"source\":\"enwiktionary\"}}\n```\n\nReverse and language-guessing rows are controlled by `include_reverse` and `include_language_guessing`.\n",
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
                wiktionary_lang: "de".to_string(),
                spelling: "schief".to_string(),
                ipa: "/ʃiːf/".to_string(),
                notation: "phonemic".to_string(),
                accent: None,
                raw_template: "{{IPA|de|/ʃiːf/}}".to_string(),
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
