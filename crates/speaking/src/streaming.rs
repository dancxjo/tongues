use serde::{Deserialize, Serialize};

use crate::segment::TerminalPunctuation;
use crate::syntax::{HeuristicLinkGrammarParser, LinkGrammarParser, SentenceSyntaxAnalysis};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StableTextChunk {
    pub text: String,
    pub terminal: Option<TerminalPunctuation>,
    pub syntax: SentenceSyntaxAnalysis,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnfinishedPrefixAnalysis {
    pub text_with_continuation: String,
    pub word_count: usize,
    pub syntax: SentenceSyntaxAnalysis,
}

#[derive(Debug, Clone)]
pub struct StableTextChunker<P = HeuristicLinkGrammarParser> {
    parser: P,
    buffer: String,
}

impl Default for StableTextChunker<HeuristicLinkGrammarParser> {
    fn default() -> Self {
        Self::new()
    }
}

impl StableTextChunker<HeuristicLinkGrammarParser> {
    pub fn new() -> Self {
        Self::with_parser(HeuristicLinkGrammarParser)
    }
}

impl<P: LinkGrammarParser> StableTextChunker<P> {
    pub fn with_parser(parser: P) -> Self {
        Self {
            parser,
            buffer: String::new(),
        }
    }

    pub fn pending_text(&self) -> &str {
        &self.buffer
    }

    pub fn push_str(&mut self, text: &str) -> Vec<StableTextChunk> {
        self.buffer.push_str(text);
        let mut released = Vec::new();

        while let Some((end, terminal)) = stable_sentence_end(&self.buffer) {
            let chunk_text = self.buffer[..end].trim().to_string();
            self.buffer.drain(..end);
            let words = words_for_parse(&chunk_text);
            let syntax = self.parser.parse(&words, terminal);
            released.push(StableTextChunk {
                text: chunk_text,
                terminal,
                syntax,
            });
        }

        released
    }

    pub fn finish(self) -> Vec<StableTextChunk> {
        let text = self.buffer.trim();
        if text.is_empty() {
            return Vec::new();
        }
        let terminal = terminal_at_end(text);
        let words = words_for_parse(text);
        let syntax = self.parser.parse(&words, terminal);
        vec![StableTextChunk {
            text: text.to_string(),
            terminal,
            syntax,
        }]
    }

    pub fn unfinished_prefix_analyses(&self) -> Vec<UnfinishedPrefixAnalysis> {
        let mut analyses = Vec::new();
        for end in word_boundary_ends(&self.buffer) {
            let prefix = self.buffer[..end].trim();
            if prefix.is_empty() || prefix.ends_with(char::is_whitespace) {
                continue;
            }
            let text_with_continuation = format!("{prefix}...");
            let words = words_for_parse(&text_with_continuation);
            let syntax = self.parser.parse(&words, None);
            analyses.push(UnfinishedPrefixAnalysis {
                text_with_continuation,
                word_count: words.len(),
                syntax,
            });
        }
        analyses
    }
}

fn stable_sentence_end(text: &str) -> Option<(usize, Option<TerminalPunctuation>)> {
    for (index, character) in text.char_indices() {
        let terminal = match character {
            '.' => TerminalPunctuation::Period,
            '?' => TerminalPunctuation::Question,
            '!' => TerminalPunctuation::Exclamation,
            _ => continue,
        };
        let end = index + character.len_utf8();
        if is_ellipsis_period(text, index) || is_decimal_or_domain_period(text, index) {
            continue;
        }
        if terminal == TerminalPunctuation::Period && period_is_provisional(text, end) {
            continue;
        }
        if has_release_evidence_after(text, end) || end == text.len() {
            return Some((end, Some(terminal)));
        }
    }
    None
}

fn has_release_evidence_after(text: &str, end: usize) -> bool {
    let after = text[end..].trim_start();
    if after.is_empty() {
        return true;
    }
    after
        .chars()
        .next()
        .is_some_and(|character| character.is_alphanumeric() || is_opening_quote(character))
}

fn period_is_provisional(text: &str, end: usize) -> bool {
    let token = token_before(text, end.saturating_sub(1));
    if token.len() == 1
        && token
            .chars()
            .all(|character| character.is_ascii_uppercase())
    {
        return true;
    }

    matches!(
        token.to_ascii_lowercase().as_str(),
        "dr" | "mr" | "mrs" | "ms" | "prof" | "sr" | "jr" | "st" | "vs"
    )
}

fn token_before(text: &str, terminal_index: usize) -> &str {
    let prefix = &text[..terminal_index];
    let end = prefix
        .char_indices()
        .rev()
        .find(|(_, character)| character.is_alphanumeric())
        .map(|(index, character)| index + character.len_utf8())
        .unwrap_or(prefix.len());
    let start = prefix[..end]
        .char_indices()
        .rev()
        .find(|(_, character)| !character.is_alphanumeric())
        .map(|(index, character)| index + character.len_utf8())
        .unwrap_or(0);
    &prefix[start..end]
}

fn is_ellipsis_period(text: &str, index: usize) -> bool {
    let bytes = text.as_bytes();
    bytes.get(index + 1) == Some(&b'.') || index > 0 && bytes.get(index - 1) == Some(&b'.')
}

fn is_decimal_or_domain_period(text: &str, index: usize) -> bool {
    let before = text[..index].chars().next_back();
    let after = text[index + 1..].chars().next();
    matches!((before, after), (Some(left), Some(right)) if left.is_alphanumeric() && right.is_alphanumeric())
}

fn terminal_at_end(text: &str) -> Option<TerminalPunctuation> {
    text.chars().rev().find_map(|character| match character {
        '.' => Some(TerminalPunctuation::Period),
        '?' => Some(TerminalPunctuation::Question),
        '!' => Some(TerminalPunctuation::Exclamation),
        _ if character.is_whitespace() || is_closing_quote(character) => None,
        _ => None,
    })
}

fn words_for_parse(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut start = None;
    for (index, character) in text.char_indices() {
        if character.is_alphanumeric() || matches!(character, '\'' | '-' | '_') {
            start.get_or_insert(index);
            continue;
        }
        if let Some(start_index) = start.take() {
            words.push(text[start_index..index].to_string());
        }
    }
    if let Some(start_index) = start {
        words.push(text[start_index..].to_string());
    }
    words
}

fn word_boundary_ends(text: &str) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut in_word = false;
    for (index, character) in text.char_indices() {
        if character.is_alphanumeric() {
            in_word = true;
        } else if in_word {
            ends.push(index);
            in_word = false;
        }
    }
    if in_word {
        ends.push(text.len());
    }
    ends
}

fn is_opening_quote(character: char) -> bool {
    matches!(character, '"' | '\'' | '(' | '[' | '{')
}

fn is_closing_quote(character: char) -> bool {
    matches!(character, '"' | '\'' | ')' | ']' | '}')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn releases_completed_sentence() {
        let mut chunker = StableTextChunker::new();
        let chunks = chunker.push_str("Okay. Next");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Okay.");
        assert_eq!(chunker.pending_text(), " Next");
    }

    #[test]
    fn does_not_release_name_initial_before_continuation() {
        let mut chunker = StableTextChunker::new();
        assert!(chunker.push_str("Who shot John F.").is_empty());

        let chunks = chunker.push_str(" Kennedy?");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Who shot John F. Kennedy?");
        assert_eq!(chunks[0].terminal, Some(TerminalPunctuation::Question));
    }

    #[test]
    fn does_not_release_honorific_before_name() {
        let mut chunker = StableTextChunker::new();
        assert!(chunker.push_str("Dr.").is_empty());

        let chunks = chunker.push_str(" King spoke.");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Dr. King spoke.");
    }

    #[test]
    fn exposes_unfinished_prefix_parse_hook() {
        let mut chunker = StableTextChunker::new();
        chunker.push_str("I want to");

        let analyses = chunker.unfinished_prefix_analyses();

        assert!(analyses.iter().any(|analysis| {
            analysis.text_with_continuation == "I want to..." && analysis.word_count == 3
        }));
    }
}
