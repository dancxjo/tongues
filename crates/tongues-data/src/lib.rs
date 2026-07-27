//! Data pipeline for tongues sequence-to-sequence translation.
//!
//! Handles CMUdict parsing, parallelized IPA phonemicization, splitting,
//! vocabulary construction, seq2seq batch collation, and shared dataset audits.

pub mod speech_corpus;

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{Context, Result};
use rand::Rng;
use rand::seq::{IndexedRandom, SliceRandom};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use speaking::{PhonemicizeRequest, VarietyId, phonemicizer_for_variety};
use tongues_core::{BOS_ID, EOS_ID, G2P_ID, P2G_ID, PAD_ID, Vocab};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OllamaVerifierConfig {
    pub family: String,
    pub model: String,
    pub url: String,
    pub rows_per_chunk: usize,
    pub max_prompt_chars: usize,
}

impl OllamaVerifierConfig {
    pub fn new(
        family: impl Into<String>,
        model: impl Into<String>,
        url: impl Into<String>,
        rows_per_chunk: usize,
        max_prompt_chars: usize,
    ) -> Self {
        Self {
            family: family.into(),
            model: model.into(),
            url: url.into(),
            rows_per_chunk,
            max_prompt_chars,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OllamaVerificationReport {
    pub model: String,
    pub url: String,
    pub rows: usize,
    #[serde(default)]
    pub total_rows: usize,
    #[serde(default)]
    pub chunks: usize,
    #[serde(default)]
    pub completed: bool,
    pub sane: bool,
    pub issue: Option<String>,
    pub raw_response: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_response_json: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunks_path: Option<PathBuf>,
    pub report_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OllamaVerificationChunkReport {
    pub model: String,
    pub url: String,
    pub chunk: usize,
    pub start_row: usize,
    pub rows: usize,
    pub sane: bool,
    pub issue: Option<String>,
    pub raw_response: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_response_json: Option<serde_json::Value>,
}

type OllamaPromptBuilder<'a, T> =
    dyn Fn(&OllamaVerifierConfig, &[T]) -> Result<(String, usize)> + 'a;

pub fn verify_jsonl_rows_with_ollama<T: Serialize>(
    config: &OllamaVerifierConfig,
    rows: &[T],
    report_path: &Path,
    chunks_path: &Path,
    prompt_builder: &OllamaPromptBuilder<'_, T>,
    mut progress: impl FnMut(usize),
) -> Result<OllamaVerificationReport> {
    anyhow::ensure!(
        config.rows_per_chunk > 0,
        "ollama rows per chunk must be greater than zero"
    );
    anyhow::ensure!(
        !config.model.trim().is_empty(),
        "ollama model must be set for {} verification",
        config.family
    );
    anyhow::ensure!(
        !config.url.trim().is_empty(),
        "ollama URL must be set for {} verification",
        config.family
    );
    anyhow::ensure!(
        !rows.is_empty(),
        "no {} training rows to verify",
        config.family
    );

    if let Some(parent) = chunks_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let chunks_part_path = part_path(chunks_path);
    let mut start = 0usize;
    let mut chunk_index = 0usize;
    let mut sane = true;
    let mut issue = None;
    let mut raw_response = String::new();
    let mut raw_response_json = None;

    if !chunks_part_path.exists() && chunks_path.exists() {
        fs::copy(chunks_path, &chunks_part_path).with_context(|| {
            format!(
                "copying existing {} to {} for resume",
                chunks_path.display(),
                chunks_part_path.display()
            )
        })?;
    }
    if chunks_part_path.exists() {
        let chunks: Vec<OllamaVerificationChunkReport> = read_jsonl_file(&chunks_part_path)?;
        let mut resumed_chunks = Vec::new();
        for mut chunk in chunks {
            anyhow::ensure!(
                chunk.model == config.model && chunk.url == config.url,
                "cannot resume {}: chunk {} was scanned with model={} url={}, current model={} url={}",
                chunks_part_path.display(),
                chunk.chunk,
                chunk.model,
                chunk.url,
                config.model,
                config.url
            );
            anyhow::ensure!(
                chunk.chunk == chunk_index && chunk.start_row == start,
                "cannot resume {}: chunk {} starts at row {}, expected chunk {} row {}",
                chunks_part_path.display(),
                chunk.chunk,
                chunk.start_row,
                chunk_index,
                start
            );
            anyhow::ensure!(
                chunk.rows > 0,
                "cannot resume {}: chunk {} has zero rows",
                chunks_part_path.display(),
                chunk.chunk
            );
            normalize_ollama_verification_chunk_report(&mut chunk);
            let next_start = start + chunk.rows;
            anyhow::ensure!(
                next_start <= rows.len(),
                "cannot resume {}: chunk {} ends at row {}, but train split has {} rows",
                chunks_part_path.display(),
                chunk.chunk,
                next_start,
                rows.len()
            );
            if is_retryable_ollama_verification_chunk(&chunk) {
                break;
            }
            if !chunk.sane {
                sane = false;
                if issue.is_none() {
                    issue = chunk.issue.clone();
                }
            }
            raw_response = chunk.raw_response.clone();
            raw_response_json = chunk.raw_response_json.clone();
            start = next_start;
            chunk_index += 1;
            resumed_chunks.push(chunk);
        }
        if resumed_chunks.len() < chunk_index || start < rows.len() {
            let mut writer = BufWriter::new(
                File::create(&chunks_part_path)
                    .with_context(|| format!("rewriting {}", chunks_part_path.display()))?,
            );
            for chunk in &resumed_chunks {
                serde_json::to_writer(&mut writer, chunk)
                    .with_context(|| format!("writing {}", chunks_part_path.display()))?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
        }
        if start > 0 {
            progress(start);
        }
    }

    let mut chunks_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&chunks_part_path)
            .with_context(|| format!("opening {}", chunks_part_path.display()))?,
    );
    while start < rows.len() {
        let end = (start + config.rows_per_chunk).min(rows.len());
        let report = verify_jsonl_chunk_with_ollama(config, &rows[start..end], prompt_builder)?;
        let scanned = report.rows.max(1).min(end - start);
        let chunk = OllamaVerificationChunkReport {
            model: report.model.clone(),
            url: report.url.clone(),
            chunk: chunk_index,
            start_row: start,
            rows: scanned,
            sane: report.sane,
            issue: report.issue.clone(),
            raw_response: report.raw_response.clone(),
            raw_response_json: report.raw_response_json.clone(),
        };
        serde_json::to_writer(&mut chunks_writer, &chunk)
            .with_context(|| format!("writing {}", chunks_part_path.display()))?;
        chunks_writer.write_all(b"\n")?;
        chunks_writer.flush()?;

        if !report.sane {
            sane = false;
            if issue.is_none() {
                issue = report.issue.clone();
            }
        }
        raw_response = report.raw_response;
        raw_response_json = report.raw_response_json;
        start += scanned;
        chunk_index += 1;
        progress(start);
    }

    chunks_writer.flush()?;
    drop(chunks_writer);
    fs::rename(&chunks_part_path, chunks_path).with_context(|| {
        format!(
            "renaming {} to {}",
            chunks_part_path.display(),
            chunks_path.display()
        )
    })?;

    let aggregate = OllamaVerificationReport {
        model: config.model.clone(),
        url: config.url.clone(),
        rows: start,
        total_rows: rows.len(),
        chunks: chunk_index,
        completed: start == rows.len(),
        sane,
        issue,
        raw_response,
        raw_response_json,
        chunks_path: Some(chunks_path.to_path_buf()),
        report_path: Some(report_path.to_path_buf()),
    };
    write_json_file_atomic(report_path, &aggregate)?;
    Ok(aggregate)
}

pub fn verify_jsonl_chunk_with_ollama<T: Serialize>(
    config: &OllamaVerifierConfig,
    rows: &[T],
    prompt_builder: &OllamaPromptBuilder<'_, T>,
) -> Result<OllamaVerificationReport> {
    let sample_rows = rows.len().min(config.rows_per_chunk);
    anyhow::ensure!(
        sample_rows > 0,
        "no {} training rows to verify",
        config.family
    );

    let (prompt, prompt_rows) = prompt_builder(config, &rows[..sample_rows])?;
    let (prompt, raw_prompt) = ollama_generate_prompt_for_model(&config.model, &prompt);
    let url = format!("{}/api/generate", config.url.trim().trim_end_matches('/'));
    let mut request = serde_json::json!({
        "model": config.model,
        "prompt": prompt,
        "stream": false,
        "think": false,
        "format": ollama_verification_response_schema(),
        "options": {
            "temperature": 0
        }
    });
    if raw_prompt {
        request["raw"] = serde_json::Value::Bool(true);
    }
    let body = serde_json::to_string(&request)?;
    let response = ureq::post(&url)
        .header("Content-Type", "application/json")
        .config()
        .http_status_as_error(false)
        .build()
        .send(body)
        .with_context(|| format!("POST {url}"))?;
    let status = response.status();
    let raw = response
        .into_body()
        .read_to_string()
        .with_context(|| format!("reading Ollama response from {url}"))?;
    anyhow::ensure!(
        status.is_success(),
        "POST {url} returned HTTP {status}: {raw}"
    );
    let generated: OllamaGenerateResponse =
        serde_json::from_str(&raw).with_context(|| format!("parsing Ollama response: {raw}"))?;
    let response_content = generated.response.trim().to_string();
    let (verifier_text, judgement, raw_response_json) =
        parse_ollama_verification_response(&response_content, &raw);
    let issue = if judgement.sane {
        None
    } else {
        Some(
            judgement
                .issue
                .filter(|issue| !issue.trim().is_empty())
                .unwrap_or_else(|| {
                    format!(
                        "Ollama reported unsane {} data without an exact issue",
                        config.family
                    )
                }),
        )
    };
    Ok(OllamaVerificationReport {
        model: config.model.clone(),
        url: config.url.clone(),
        rows: prompt_rows,
        total_rows: prompt_rows,
        chunks: 1,
        completed: true,
        sane: judgement.sane,
        issue,
        raw_response: verifier_text,
        raw_response_json,
        chunks_path: None,
        report_path: None,
    })
}

fn part_path(path: &Path) -> PathBuf {
    path.with_extension(format!(
        "{}part",
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ))
}

fn is_retryable_ollama_verification_chunk(chunk: &OllamaVerificationChunkReport) -> bool {
    chunk.issue.as_deref().is_some_and(|issue| {
        !chunk.sane
            && (issue.starts_with("verifier response did not match expected schema:")
                || is_unactionable_ollama_verification_issue(issue))
    })
}

fn normalize_ollama_verification_chunk_report(chunk: &mut OllamaVerificationChunkReport) {
    if let Some(value) = chunk.raw_response_json.clone() {
        if let Ok(mut judgement) = serde_json::from_value::<OllamaVerificationJudgement>(value) {
            normalize_ollama_verification_judgement(&mut judgement);
            chunk.sane = judgement.sane;
            chunk.issue = if judgement.sane {
                None
            } else {
                judgement.issue.filter(|issue| !issue.trim().is_empty())
            };
            return;
        }
    }
    if let Ok(judgement) = parse_ollama_verification_judgement(&chunk.raw_response) {
        chunk.sane = judgement.sane;
        chunk.issue = if judgement.sane {
            None
        } else {
            judgement.issue.filter(|issue| !issue.trim().is_empty())
        };
    }
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    #[serde(default)]
    response: String,
}

#[derive(Debug, Deserialize)]
struct OllamaVerificationJudgement {
    sane: bool,
    #[serde(default)]
    issue: Option<String>,
}

fn ollama_verification_response_schema() -> serde_json::Value {
    serde_json::json!({
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "sane": { "const": true },
                    "issue": { "type": "null" }
                },
                "required": ["sane", "issue"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "sane": { "const": false },
                    "issue": {
                        "type": "string",
                        "minLength": 32,
                        "maxLength": 160
                    }
                },
                "required": ["sane", "issue"],
                "additionalProperties": false
            }
        ]
    })
}

fn ollama_generate_prompt_for_model(model: &str, prompt: &str) -> (String, bool) {
    if is_gpt_oss_ollama_model(model) {
        (
            format!(
                "<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>"
            ),
            true,
        )
    } else {
        (prompt.to_string(), false)
    }
}

fn is_gpt_oss_ollama_model(model: &str) -> bool {
    let model = model.trim();
    model == "gpt-oss" || model.starts_with("gpt-oss:")
}

fn parse_ollama_verification_response(
    content: &str,
    raw: &str,
) -> (
    String,
    OllamaVerificationJudgement,
    Option<serde_json::Value>,
) {
    let candidates = [content.trim()];
    let mut last_error = None;
    for candidate in candidates {
        if candidate.is_empty() {
            continue;
        }
        match parse_ollama_verification_judgement(candidate) {
            Ok(judgement) => {
                let raw_response_json = extract_ollama_verification_json(candidate)
                    .ok()
                    .and_then(|json| serde_json::from_str(&json).ok());
                return (candidate.to_string(), judgement, raw_response_json);
            }
            Err(error) => last_error = Some(error),
        }
    }
    let fallback = if !content.trim().is_empty() {
        content.trim()
    } else {
        raw.trim()
    };
    let detail = last_error
        .map(|error| error.to_string())
        .unwrap_or_else(|| "Ollama returned empty verifier content".to_string());
    let detail = detail
        .strip_prefix("parsing verifier judgement: ")
        .unwrap_or(&detail)
        .to_string();
    (
        fallback.to_string(),
        OllamaVerificationJudgement {
            sane: false,
            issue: Some(format!(
                "verifier response did not match expected schema: {detail}"
            )),
        },
        None,
    )
}

fn parse_ollama_verification_judgement(raw: &str) -> Result<OllamaVerificationJudgement> {
    let json = extract_ollama_verification_json(raw)?;
    let value: serde_json::Value =
        serde_json::from_str(&json).with_context(|| format!("parsing verifier JSON: {raw}"))?;
    let mut judgement: OllamaVerificationJudgement = serde_json::from_value(value)
        .with_context(|| format!("parsing verifier judgement: {raw}"))?;
    normalize_ollama_verification_judgement(&mut judgement);
    Ok(judgement)
}

fn extract_ollama_verification_json(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    anyhow::ensure!(
        !trimmed.is_empty(),
        "Ollama returned empty verifier content"
    );
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return Ok(trimmed.to_string());
    }
    for (start, character) in trimmed.char_indices() {
        if character != '{' {
            continue;
        }
        if let Some(end) = json_object_end(trimmed, start) {
            let candidate = &trimmed[start..end];
            if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                return Ok(candidate.to_string());
            }
        }
    }
    anyhow::bail!("parsing verifier JSON: {raw}");
}

fn json_object_end(raw: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, character) in raw[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        match character {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(start + offset + character.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

fn normalize_ollama_verification_judgement(judgement: &mut OllamaVerificationJudgement) {
    if judgement.sane {
        if judgement
            .issue
            .as_deref()
            .is_some_and(|issue| !issue.trim().is_empty())
        {
            judgement.sane = false;
            judgement.issue = Some(
                "verifier response did not match expected schema: sane=true requires issue=null"
                    .to_string(),
            );
        } else {
            judgement.issue = None;
        }
    } else if judgement
        .issue
        .as_deref()
        .is_some_and(is_unactionable_ollama_verification_issue)
    {
        judgement.issue = Some(
            "verifier response did not match expected schema: issue is not actionable".to_string(),
        );
    }
}

fn is_unactionable_ollama_verification_issue(issue: &str) -> bool {
    let trimmed = issue.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_lowercase();
    let normalized = lower
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { '-' })
        .collect::<String>();
    let normalized = normalized.trim_matches('-');
    let compact = normalized.replace('-', "");
    if is_bare_audit_reference(&compact) {
        return true;
    }
    let has_row_reference = lower.contains("audit_row")
        || lower.contains("audit-row")
        || lower.contains("audit row")
        || lower.contains("audit-row")
        || lower.contains("row ");
    if has_row_reference && is_bare_row_reference(&compact) {
        return true;
    }
    if has_row_reference && looks_like_known_head2phones_false_positive(&lower, &compact) {
        return true;
    }
    if has_row_reference {
        return false;
    }
    if looks_like_known_head2phones_false_positive(&lower, &compact) {
        return true;
    }
    matches!(
        normalized,
        "audit"
            | "data"
            | "issue-001"
            | "format-check"
            | "head-split-format-check"
            | "audit-output-format-error"
            | "head-text-format-check"
            | "head-length-mismatch"
            | "head-found-sanity-check"
            | "head-output-sanity"
            | "lang-mismatch"
            | "head-not-found"
    )
}

fn is_bare_audit_reference(compact: &str) -> bool {
    compact
        .strip_prefix("audit")
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()))
}

fn is_bare_row_reference(compact: &str) -> bool {
    compact
        .strip_prefix("auditrow")
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()))
        || compact
            .strip_prefix("row")
            .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()))
}

fn looks_like_known_head2phones_false_positive(lower: &str, compact: &str) -> bool {
    let says_no_issue = lower.contains("no issue")
        || lower.contains("no issues")
        || lower.contains("no problem")
        || lower.contains("none found")
        || lower.contains("none detected")
        || lower.contains("no problems found")
        || lower.contains("none of the rows violate")
        || lower.contains("all rows are consistent")
        || lower.contains("all rows are valid")
        || lower.contains("correctly formatted")
        || lower.contains("consistent with the expected")
        || lower.contains("conform to the specification")
        || lower.contains("conform to the expected")
        || lower.contains("valid according to the specification")
        || lower.contains("dataset is consistent")
        || lower.contains("dataset appears consistent");
    let mentions_invented_split = compact.contains("missingrequiredfieldsplit")
        || compact.contains("splittag")
        || compact.contains("splitpoint")
        || compact.contains("splitaftertag")
        || compact.contains("missingsplitafter");
    let mentions_invented_phone_marker =
        compact.contains("missingphonemes") || compact.contains("missingphonemen");
    let mentions_no_head_split_zero = compact.contains("nohead")
        && (compact.contains("splitafterof0") || compact.contains("split0"));
    let mentions_missing_head_split =
        compact.contains("headismissing") && compact.contains("splitispresent");
    let mentions_verifier_contract = lower.contains("audit_id")
        || lower.contains("is_sane")
        || lower.contains("is_valid_output_tags")
        || lower.contains("the json is truncated")
        || lower.contains("the user wants");
    let mentions_lang_mismatch_example = compact.contains("langmismatch")
        && compact.contains("contains")
        && (compact.contains("phones") || compact.contains("phonemes"));
    let mentions_grapheme_recalculation = compact.contains("headlength")
        && (compact.contains("splitafter") || compact.contains("splitlength"))
        && (compact.contains("matches")
            || compact.contains("graphemes")
            || compact.contains("characters")
            || compact.contains("stringlength"));
    let mentions_random_cut_prefix_as_head = (compact.contains("nohead")
        || compact.contains("noheadfound")
        || compact.contains("missinghead"))
        && (compact.contains("singlecharacter")
            || compact.contains("completeword")
            || compact.contains("validstartofaword")
            || compact.contains("nonemptyinput")
            || compact.contains("partialword")
            || compact.contains("wordfragment")
            || compact.contains("longerhead")
            || compact.contains("couldbeapartofalongerhead")
            || compact.contains("couldformahead"));
    let mentions_structural_language_span_false_positive = compact.contains("languagespans")
        && (compact.contains("missingphones")
            || compact.contains("missingphonetic")
            || compact.contains("phonesblock")
            || compact.contains("phonestag")
            || compact.contains("missingdetectedlang")
            || compact.contains("detectedlangtagisnotallowed")
            || compact.contains("detectedlangshouldonlyappear"));
    let mentions_detected_lang_forbidden = compact.contains("detectedlang")
        && (compact.contains("shouldonlyappear") || compact.contains("isnotallowed"));
    let mentions_transcription_quality = compact.contains("phonetictranscription")
        || compact.contains("phoneticvariants")
        || compact.contains("pronunciation")
        || compact.contains("invalidipasymbol")
        || compact.contains("validipasymbol")
        || compact.contains("diacritic")
        || compact.contains("wrongvowel")
        || compact.contains("dialects")
        || compact.contains("varieties");
    let looks_like_placeholder = compact.starts_with("headlengthmismatch")
        || compact.starts_with("auditrow")
            && (compact.contains("headmismatch")
                || compact.contains("missinghead")
                || compact.contains("splitmismatch"))
        || compact.starts_with("auditissue")
        || compact.starts_with("audit")
            && compact.chars().filter(|ch| ch.is_ascii_digit()).count() > 8
        || compact.starts_with("noneissueidentified")
        || compact.contains("111111")
        || compact.starts_with("auditfailed")
        || compact.starts_with("transcriptionerror");

    says_no_issue
        || mentions_invented_split
        || mentions_invented_phone_marker
        || mentions_no_head_split_zero
        || mentions_missing_head_split
        || mentions_verifier_contract
        || mentions_lang_mismatch_example
        || mentions_grapheme_recalculation
        || mentions_random_cut_prefix_as_head
        || mentions_structural_language_span_false_positive
        || mentions_detected_lang_forbidden
        || mentions_transcription_quality
        || looks_like_placeholder
}

fn read_jsonl_file<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str(&line).with_context(|| {
                format!("parsing JSONL row {} in {}", index + 1, path.display())
            })?,
        );
    }
    Ok(rows)
}

fn write_json_file_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let part = part_path(path);
    fs::write(&part, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("writing {}", part.display()))?;
    fs::rename(&part, path)
        .with_context(|| format!("renaming {} to {}", part.display(), path.display()))
}

// ── Lexeme ─────────────────────────────────────────────────────────────────

/// Multimodal pronunciation entry storing spelling, broad IPA phonemes, and narrow IPA phones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lexeme {
    /// Orthographic spelling of the base word.
    pub base_word: String,
    /// Broad IPA phoneme string.
    pub phonemes: String,
    /// 0-indexed OpenEPD/wordfreq rarity rank; lower means more frequent.
    pub rarity: f32,
}

// ── CMUdict parsing and parallel IPA generation ────────────────────────────

/// Parse base words from a CMUdict `.dict` file, keeping only standard alphabetical words.
pub fn parse_cmudict(text: &str) -> Vec<String> {
    let mut base_words = std::collections::BTreeSet::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(";;;") {
            continue;
        }
        let mut tokens = line.split_ascii_whitespace();
        let raw_word = match tokens.next() {
            Some(w) => w,
            None => continue,
        };

        // Extract base word by removing alternate suffix like "(2)"
        let base_word = if let Some(open) = raw_word.find('(') {
            raw_word[..open].to_lowercase()
        } else {
            raw_word.to_lowercase()
        };

        // Only keep alphabetical base words with optional apostrophes/hyphens
        if !base_word.is_empty()
            && base_word
                .chars()
                .all(|c| c.is_alphabetic() || c == '\'' || c == '-')
        {
            base_words.insert(base_word);
        }
    }
    base_words.into_iter().collect()
}

/// Phonemicize a single base word into its broad and narrow IPA string representations.
pub fn phonemicize_word(base_word: &str) -> Option<(String, String)> {
    let variety = VarietyId("en-US".to_string());
    let phonemicizer = phonemicizer_for_variety(&variety).ok()?;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: base_word.to_string(),
            variety,
            style: None,
        })
        .ok()?;

    if phonemicized
        .warnings
        .iter()
        .any(|w| w.kind == speaking::PronunciationWarningKind::GuessedWord)
    {
        return None;
    }

    let plan = speaking::UtterancePlan::from(&phonemicized);
    let broad = speaking::display_plan_phoneme_words(&plan);
    let narrow = speaking::display_plan_phone_words(&plan);
    if broad.is_empty() || narrow.is_empty() {
        None
    } else {
        Some((broad, narrow))
    }
}

/// Run multi-threaded parallel IPA phonemicization for a list of base words.
pub fn phonemicize_lexemes(base_words: Vec<String>) -> Vec<Lexeme> {
    let base_words = Arc::new(base_words);
    let results = Arc::new(Mutex::new(Vec::new()));
    let num_threads = 20;
    let mut handles = Vec::new();

    let chunk_size = base_words.len().div_ceil(num_threads);

    for t in 0..num_threads {
        let base_words = Arc::clone(&base_words);
        let results = Arc::clone(&results);
        let start_idx = t * chunk_size;
        let end_idx = (start_idx + chunk_size).min(base_words.len());

        if start_idx >= base_words.len() {
            break;
        }

        let handle = thread::spawn(move || {
            let mut local_results = Vec::new();
            for i in start_idx..end_idx {
                let word = &base_words[i];
                if let Some((phonemes, _phones)) = phonemicize_word(word) {
                    local_results.push(Lexeme {
                        base_word: word.clone(),
                        phonemes,
                        rarity: 50_000.0,
                    });
                }
            }
            let mut guard = results.lock().unwrap();
            guard.extend(local_results);
        });
        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }

    let guard = results.lock().unwrap();
    guard.clone()
}

// ── Vocabulary builder ─────────────────────────────────────────────────────

/// Build the full unified vocabulary from a collection of lexemes.
pub fn build_vocab(lexemes: &[Lexeme]) -> Vocab {
    let mut words = Vec::new();
    let mut phonemes = Vec::new();

    for lex in lexemes {
        words.push(lex.base_word.clone());
        phonemes.push(lex.phonemes.clone());
    }

    Vocab::build(&words, &phonemes, &[])
}

// ── Data splitting ─────────────────────────────────────────────────────────

/// Split lexemes into train / valid / test sets.
pub fn split_by_base_word<R: Rng>(
    lexemes: &[Lexeme],
    train_frac: f64,
    valid_frac: f64,
    rng: &mut R,
) -> (Vec<Lexeme>, Vec<Lexeme>, Vec<Lexeme>) {
    let mut lexemes = lexemes.to_vec();
    lexemes.shuffle(rng);

    let n = lexemes.len();
    let train_end = (n as f64 * train_frac).round() as usize;
    let valid_end = train_end + (n as f64 * valid_frac).round() as usize;

    let mut train = Vec::new();
    let mut valid = Vec::new();
    let mut test = Vec::new();

    for (i, lex) in lexemes.into_iter().enumerate() {
        if i < train_end {
            train.push(lex);
        } else if i < valid_end {
            valid.push(lex);
        } else {
            test.push(lex);
        }
    }

    (train, valid, test)
}

/// Verify that no group identity appears in more than one named split.
///
/// Dataset families should pass their stable pre-expansion group identity
/// (source recording, lexical entry, fixture, session, and so on), rather than
/// a derived row identifier.
pub fn check_group_split_leakage(splits: &[(&str, Vec<String>)]) -> Vec<String> {
    use std::collections::{BTreeMap, BTreeSet};

    let mut seen: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (split, groups) in splits {
        for group in groups {
            seen.entry(group.clone())
                .or_default()
                .insert((*split).to_string());
        }
    }

    seen.into_iter()
        .filter_map(|(group, splits)| {
            (splits.len() > 1).then(|| {
                let split_list = splits.into_iter().collect::<Vec<_>>().join(", ");
                format!("group `{group}` appears in multiple splits: {split_list}")
            })
        })
        .collect()
}

/// Verify that no base-word group appears in more than one split.
pub fn check_split_leakage(train: &[Lexeme], valid: &[Lexeme], test: &[Lexeme]) -> Vec<String> {
    check_group_split_leakage(&[
        (
            "train",
            train
                .iter()
                .map(|lexeme| lexeme.base_word.clone())
                .collect(),
        ),
        (
            "valid",
            valid
                .iter()
                .map(|lexeme| lexeme.base_word.clone())
                .collect(),
        ),
        (
            "test",
            test.iter().map(|lexeme| lexeme.base_word.clone()).collect(),
        ),
    ])
}

// ── Seq2Seq Task Representation & Collation ────────────────────────────────

/// Available translation directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Task {
    G2P,
    P2G,
}

impl Task {
    /// Get the vocabulary ID corresponding to this task's prefix token.
    pub fn get_prefix_id(&self) -> u32 {
        match self {
            Task::G2P => G2P_ID,
            Task::P2G => P2G_ID,
        }
    }

    /// Randomly sample a task from all available tasks.
    pub fn sample<R: Rng>(rng: &mut R) -> Self {
        let tasks = [Task::G2P, Task::P2G];
        *tasks.choose(rng).unwrap()
    }

    /// Parse a task direction from a string slice.
    // Retain the existing optional parser API used by callers; changing this
    // to FromStr would alter both its return type and compatibility contract.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "g2p" => Some(Task::G2P),
            "p2g" => Some(Task::P2G),
            _ => None,
        }
    }
}

/// A single translation training example.
#[derive(Debug, Clone)]
pub struct Seq2SeqExample {
    /// Token IDs for source sequence (starts with Task Token).
    pub src_ids: Vec<u32>,
    /// Token IDs for target decoder input (starts with BOS).
    pub tgt_in_ids: Vec<u32>,
    /// Token IDs for target decoder loss output (ends with EOS).
    pub tgt_out_ids: Vec<u32>,
}

/// Convert a Lexeme to a translation example.
pub fn make_seq2seq_example(lexeme: &Lexeme, task: Task, vocab: &Vocab) -> Seq2SeqExample {
    let base_word = lexeme.base_word.to_lowercase();
    let (src_str, tgt_str) = match task {
        Task::G2P => (base_word.as_str(), lexeme.phonemes.as_str()),
        Task::P2G => (lexeme.phonemes.as_str(), base_word.as_str()),
    };

    let mut src_ids = vec![task.get_prefix_id()];
    src_ids.extend(vocab.encode_string(src_str));

    let mut tgt_in_ids = vec![BOS_ID];
    tgt_in_ids.extend(vocab.encode_string(tgt_str));

    let mut tgt_out_ids = vocab.encode_string(tgt_str);
    tgt_out_ids.push(EOS_ID);

    Seq2SeqExample {
        src_ids,
        tgt_in_ids,
        tgt_out_ids,
    }
}

/// Padded batch ready for the sequence-to-sequence model.
#[derive(Debug, Clone)]
pub struct Batch {
    /// `[batch, max_src_len]` source token IDs.
    pub src_ids: Vec<Vec<i32>>,
    /// `[batch, max_tgt_len]` target input token IDs.
    pub tgt_in_ids: Vec<Vec<i32>>,
    /// `[batch, max_tgt_len]` target output token IDs.
    pub tgt_out_ids: Vec<Vec<i32>>,
    /// `[batch, max_src_len]` padding mask (true for padding).
    pub src_pad_mask: Vec<Vec<bool>>,
    /// `[batch, max_tgt_len]` padding mask (true for padding).
    pub tgt_pad_mask: Vec<Vec<bool>>,
    /// Number of examples in the batch.
    pub size: usize,
}

/// Collate sequence-to-sequence examples into a padded batch.
pub fn collate_batch(examples: &[Seq2SeqExample], max_src_len: usize, max_tgt_len: usize) -> Batch {
    let size = examples.len();
    let mut src_ids = vec![vec![PAD_ID as i32; max_src_len]; size];
    let mut tgt_in_ids = vec![vec![PAD_ID as i32; max_tgt_len]; size];
    let mut tgt_out_ids = vec![vec![PAD_ID as i32; max_tgt_len]; size];

    let mut src_pad_mask = vec![vec![true; max_src_len]; size];
    let mut tgt_pad_mask = vec![vec![true; max_tgt_len]; size];

    for (i, ex) in examples.iter().enumerate() {
        for (j, &id) in ex.src_ids.iter().enumerate().take(max_src_len) {
            src_ids[i][j] = id as i32;
            src_pad_mask[i][j] = false;
        }
        for (j, &id) in ex.tgt_in_ids.iter().enumerate().take(max_tgt_len) {
            tgt_in_ids[i][j] = id as i32;
            tgt_pad_mask[i][j] = false;
        }
        for (j, &id) in ex.tgt_out_ids.iter().enumerate().take(max_tgt_len) {
            tgt_out_ids[i][j] = id as i32;
        }
    }

    Batch {
        src_ids,
        tgt_in_ids,
        tgt_out_ids,
        src_pad_mask,
        tgt_pad_mask,
        size,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_cmu_parsing_base_words() {
        let text = ";;; comment\nHELLO H EH1 L OW0\nWORLD(2) W ER1 L D\n12345 NOPE\n";
        let base_words = parse_cmudict(text);
        assert_eq!(base_words, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn lexeme_pronunciations_are_views_of_the_speaking_plan() {
        let (broad, narrow) = phonemicize_word("atlas").expect("known pronunciation");
        assert_eq!(broad, "ˈæt.ləs");
        assert_eq!(narrow, "ˈæt.ləs");
    }

    #[test]
    fn test_split_no_leakage() {
        let lex = vec![
            Lexeme {
                base_word: "cat".into(),
                phonemes: "kæt".into(),
                rarity: 2_000.0,
            },
            Lexeme {
                base_word: "dog".into(),
                phonemes: "dɔɡ".into(),
                rarity: 1_000.0,
            },
        ];
        let mut rng = StdRng::seed_from_u64(42);
        let (train, valid, test) = split_by_base_word(&lex, 0.5, 0.5, &mut rng);
        assert_eq!(train.len() + valid.len() + test.len(), 2);
        assert!(check_split_leakage(&train, &valid, &test).is_empty());
    }

    #[test]
    fn test_split_leakage_reports_conflicting_groups() {
        let train = vec![Lexeme {
            base_word: "cat".into(),
            phonemes: "kæt".into(),
            rarity: 1_000.0,
        }];
        let valid = vec![Lexeme {
            base_word: "cat".into(),
            phonemes: "kæt".into(),
            rarity: 1_000.0,
        }];
        let conflicts = check_split_leakage(&train, &valid, &[]);
        assert_eq!(conflicts.len(), 1);
        assert!(conflicts[0].contains("cat"));
        assert!(conflicts[0].contains("train"));
        assert!(conflicts[0].contains("valid"));
    }

    #[test]
    fn seq2seq_examples_normalize_spelling_to_lowercase() {
        let lex = Lexeme {
            base_word: "FARKLE".into(),
            phonemes: "ˈfɑɹ.kəl".into(),
            rarity: 50_000.0,
        };
        let vocab = Vocab::build(&["farkle".to_string()], &["ˈfɑɹ.kəl".to_string()], &[]);

        let g2p = make_seq2seq_example(&lex, Task::G2P, &vocab);
        let p2g = make_seq2seq_example(&lex, Task::P2G, &vocab);

        assert_eq!(vocab.decode_ids(&g2p.src_ids[1..]), "farkle");
        assert_eq!(vocab.decode_ids(&p2g.tgt_out_ids), "farkle");
    }

    #[test]
    fn lexeme_json_requires_rarity() {
        let err = serde_json::from_str::<Lexeme>(r#"{"base_word":"cat","phonemes":"kæt"}"#)
            .expect_err("rarity should be required");

        assert!(err.to_string().contains("missing field `rarity`"));
    }

    #[test]
    fn ollama_verification_rejects_bare_audit_references() {
        for raw in [
            r#"{"issue":"audit-1","sane":false}"#,
            r#"{"issue":"audit_row 18","sane":false}"#,
            r#"{"issue":"row 18","sane":false}"#,
            r#"{"issue":"none found, all rows are consistent","sane":false}"#,
            r#"{"issue":"No issues found. The dataset is consistent with the specification.","sane":false}"#,
            r#"{"issue":"head_length_mismatch_1_2_3_4_5_6","sane":false}"#,
            r#"{"issue":"audit_row 6: head length 48 but split_after 48, but head contains 48 graphemes? No issue.","sane":false}"#,
            r#"{"issue":"audit_failed_1_1_1_1_1_1","sane":false}"#,
            r#"{"issue":"transcription_error_1_1_1_1_1_1","sane":false}"#,
            r#"{"issue":"No problem found in the supplied rows","sane":false}"#,
            r#"{"issue":"All rows are valid according to the specification. No problems detected.","sane":false}"#,
            r#"{"issue":"No head found for a partial word that could be part of a longer head","sane":false}"#,
            r#"{"issue":"No head for a single-character input that is a valid start of a word","sane":false}"#,
            r#"{"issue":"synthesized language-spans block missing a <PHONES> block","sane":false}"#,
            r#"{"issue":"The <HEAD_FOUND> block contains a <DETECTED_LANG> tag that should only appear elsewhere","sane":false}"#,
            r#"{"issue":"Inconsistent phonetic transcription for the same head across different varieties","sane":false}"#,
            r#"{"issue":"PHONES contains a diacritic that is not a valid IPA symbol","sane":false}"#,
            r#"{"issue":"audit_issue_1_1_1_1_1_1_1_1_1_1","sane":false}"#,
        ] {
            let judgement = parse_ollama_verification_judgement(raw)
                .expect("bare audit reference should parse as verifier failure");
            assert!(!judgement.sane);
            assert_eq!(
                judgement.issue.as_deref(),
                Some("verifier response did not match expected schema: issue is not actionable")
            );
        }
    }

    #[test]
    fn ollama_verification_rejects_sane_with_non_null_issue() {
        let judgement =
            parse_ollama_verification_judgement(r#"{"issue":"No issues found","sane":true}"#)
                .expect("schema-shaped judgement should parse as verifier failure");
        assert!(!judgement.sane);
        assert_eq!(
            judgement.issue.as_deref(),
            Some("verifier response did not match expected schema: sane=true requires issue=null")
        );
    }
}
