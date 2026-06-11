//! Sentence-parser model-family scaffold.
//!
//! The current implementation preserves the artifact and output contracts for
//! a future neural parser while delegating parsing to the existing rule-based
//! `speech::syntax` parser.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use speech::segment::TerminalPunctuation;
use speech::syntax::{HeuristicLinkGrammarParser, LinkGrammarParser, SentenceSyntaxAnalysis};
use tongues_neural::{write_manifest, ModelArtifactManifest};

pub const FAMILY: &str = "sentence-parser";
pub const ARCHITECTURE: &str = "heuristic-contract-scaffold";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SentenceParserConfig {
    pub dataset_id: String,
    pub lowercase: bool,
}

impl Default for SentenceParserConfig {
    fn default() -> Self {
        Self {
            dataset_id: "v0".to_string(),
            lowercase: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerSpec {
    pub kind: String,
    pub lowercase: bool,
}

impl Default for TokenizerSpec {
    fn default() -> Self {
        Self {
            kind: "whitespace".to_string(),
            lowercase: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelSchema {
    pub output_type: String,
}

impl Default for LabelSchema {
    fn default() -> Self {
        Self {
            output_type: "speech::syntax::SentenceSyntaxAnalysis".to_string(),
        }
    }
}

pub fn prepare_dataset(out: &Path, config: &SentenceParserConfig) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    fs::write(
        out.join("dataset_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        out.join("README.md"),
        "Sentence parser dataset scaffold. Add train/valid/test JSONL data here.\n",
    )?;
    Ok(())
}

pub fn write_scaffold_model(out: &Path, config: &SentenceParserConfig) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let tokenizer = TokenizerSpec {
        lowercase: config.lowercase,
        ..TokenizerSpec::default()
    };
    fs::write(out.join("model.bin"), b"sentence-parser-scaffold\n")?;
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
    fs::write(
        out.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer)?,
    )?;
    fs::write(
        out.join("label_schema.json"),
        serde_json::to_string_pretty(&LabelSchema::default())?,
    )?;
    write_manifest(
        out,
        &ModelArtifactManifest::new(FAMILY, ARCHITECTURE, &config.dataset_id),
    )
}

pub fn parse_sentence(text: &str, lowercase: bool) -> SentenceSyntaxAnalysis {
    let mut words = text
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| matches!(c, '.' | '?' | '!' | ',' | ';' | ':'))
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if lowercase {
        words = words.into_iter().map(|word| word.to_lowercase()).collect();
    }
    let terminal = terminal_from_text(text);
    HeuristicLinkGrammarParser.parse(&words, terminal)
}

fn terminal_from_text(text: &str) -> Option<TerminalPunctuation> {
    match text.trim_end().chars().last()? {
        '?' => Some(TerminalPunctuation::Question),
        '!' => Some(TerminalPunctuation::Exclamation),
        '.' => Some(TerminalPunctuation::Period),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_output_matches_speech_syntax_contract() {
        let analysis = parse_sentence("The quick brown fox jumps.", false);
        let raw = serde_json::to_string(&analysis).unwrap();
        let reparsed: SentenceSyntaxAnalysis = serde_json::from_str(&raw).unwrap();

        assert_eq!(reparsed.terminal, Some(TerminalPunctuation::Period));
        assert!(!reparsed.tokens.is_empty());
    }
}
