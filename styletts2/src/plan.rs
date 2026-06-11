use serde::{Deserialize, Serialize};
use speech::{TerminalPunctuation, UtteranceId, UtterancePlan, VarietyId};

use crate::backend::StyleTts2Error;
use crate::symbols::{
    StyleTts2SymbolMapper, StyleTts2SymbolSource, StyleTts2SymbolToken, SymbolSet,
    styletts2_en_us_symbol_set,
};

pub const DEFAULT_MAX_TTS_SYMBOLS: usize = 180;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendSynthesisPlan {
    pub utterance_id: UtteranceId,
    pub variety: VarietyId,
    pub text: Option<String>,
    pub chunks: Vec<SynthesisChunk>,
    pub max_symbols_per_chunk: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynthesisChunk {
    pub symbols: Vec<StyleTts2SymbolToken>,
    pub terminal: Option<TerminalPunctuation>,
    pub source_text: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StyleTts2PlanOptions {
    pub max_symbols_per_chunk: usize,
    pub chunking_enabled: bool,
}

impl Default for StyleTts2PlanOptions {
    fn default() -> Self {
        Self {
            max_symbols_per_chunk: DEFAULT_MAX_TTS_SYMBOLS,
            chunking_enabled: true,
        }
    }
}

pub fn prepare_styletts2_plan(
    utterance_plan: &UtterancePlan,
    symbol_set: &SymbolSet,
    options: StyleTts2PlanOptions,
) -> Result<BackendSynthesisPlan, StyleTts2Error> {
    let sequence = symbol_set.lower(utterance_plan)?;
    let max_symbols_per_chunk = options.max_symbols_per_chunk.max(1);
    let mut chunks = chunk_symbols(
        sequence.tokens,
        max_symbols_per_chunk,
        options.chunking_enabled,
    );
    if chunks.len() == 1 {
        chunks[0].source_text = utterance_plan.intended_text.clone();
    }
    Ok(BackendSynthesisPlan {
        utterance_id: utterance_plan.id.clone(),
        variety: utterance_plan.variety.clone(),
        text: utterance_plan.intended_text.clone(),
        chunks,
        max_symbols_per_chunk,
    })
}

pub fn validate_styletts2_plan(plan: &BackendSynthesisPlan) -> Result<(), StyleTts2Error> {
    let symbol_set = styletts2_en_us_symbol_set();
    if plan.chunks.is_empty() {
        return Err(invalid_output(
            "StyleTTS2 synthesis plan has no non-empty chunks",
        ));
    }
    if plan.max_symbols_per_chunk == 0 {
        return Err(invalid_output(
            "StyleTTS2 max_symbols_per_chunk must be greater than zero",
        ));
    }

    for (chunk_index, chunk) in plan.chunks.iter().enumerate() {
        let display_index = chunk_index + 1;
        if chunk.symbols.is_empty() {
            return Err(invalid_output(format!(
                "StyleTTS2 synthesis chunk {display_index} is empty"
            )));
        }
        if chunk.symbols.len() > plan.max_symbols_per_chunk {
            return Err(invalid_output(format!(
                "StyleTTS2 synthesis chunk {display_index} has {} symbols, above max {}",
                chunk.symbols.len(),
                plan.max_symbols_per_chunk
            )));
        }
        for token in &chunk.symbols {
            if !symbol_set.symbols.contains(&token.symbol) {
                return Err(invalid_output(format!(
                    "unknown StyleTTS2 symbol `{}` in synthesis chunk {display_index}",
                    token.symbol
                )));
            }
            let text = styletts2_text_for_symbol(&token.symbol)?;
            for character in text.chars() {
                if styletts2_character_id(character).is_none() {
                    return Err(invalid_output(format!(
                        "StyleTTS2 text-cleaner vocabulary has no token for `{character}`"
                    )));
                }
            }
        }
    }

    Ok(())
}

pub fn styletts2_token_ids_for_symbols(
    symbols: &[StyleTts2SymbolToken],
) -> Result<Vec<i64>, StyleTts2Error> {
    let text = styletts2_text_for_symbols(symbols)?;
    let text = text.trim();
    styletts2_text_to_ids(text)
}

pub fn styletts2_text_to_ids(styletts2_text: &str) -> Result<Vec<i64>, StyleTts2Error> {
    let text = styletts2_text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let mut ids = Vec::with_capacity(text.chars().count() + 2);
    ids.push(0);
    for character in text.chars() {
        let id = styletts2_character_id(character).ok_or_else(|| {
            invalid_output(format!(
                "StyleTTS2 text-cleaner vocabulary has no token for `{character}`"
            ))
        })?;
        ids.push(id);
    }
    Ok(ids)
}

pub fn styletts2_text_for_symbols(
    symbols: &[StyleTts2SymbolToken],
) -> Result<String, StyleTts2Error> {
    let mut text = String::new();
    for token in symbols {
        text.push_str(styletts2_text_for_symbol(&token.symbol)?);
    }
    Ok(text)
}

pub fn styletts2_text_for_symbol(symbol: &str) -> Result<&'static str, StyleTts2Error> {
    let text = match symbol {
        "AA" => "ɑː",
        "AE" => "æ",
        "AH" => "ə",
        "AO" => "ɔː",
        "AW" => "aʊ",
        "AY" => "aɪ",
        "B" => "b",
        "CH" => "tʃ",
        "D" => "d",
        "DH" => "ð",
        "EH" => "ɛ",
        "ER" => "ɝ",
        "EY" => "eɪ",
        "F" => "f",
        "G" => "ɡ",
        "HH" => "h",
        "IH" => "ɪ",
        "IY" => "iː",
        "JH" => "dʒ",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ",
        "OW" => "oʊ",
        "OY" => "ɔɪ",
        "P" => "p",
        "R" => "ɹ",
        "S" => "s",
        "SH" => "ʃ",
        "T" => "t",
        "TH" => "θ",
        "UH" => "ʊ",
        "UW" => "uː",
        "V" => "v",
        "W" => "w",
        "Y" => "j",
        "Z" => "z",
        "ZH" => "ʒ",
        "ə" => "ə",
        "ɐ" => "ɐ",
        "ʌ" => "ʌ",
        "ɚ" => "ɚ",
        "ɝ" => "ɝ",
        "ɫ" => "l",
        "|" => " ",
        "." => " . ",
        "!" => " ! ",
        "?" => " ? ",
        "," => " , ",
        ";" => " ; ",
        ":" => " : ",
        "ˈ" => "ˈ",
        "ˌ" => "ˌ",
        "↗" => "↗",
        "↘" => "↘",
        "→" => "→",
        _ => {
            return Err(invalid_output(format!(
                "cannot map lowered StyleTTS2 symbol `{symbol}` to text-cleaner input"
            )));
        }
    };
    Ok(text)
}

pub fn styletts2_character_id(character: char) -> Option<i64> {
    const SYMBOLS: &str = "$;:,.!?¡¿—…\"«»“” ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘̩ᵻ";
    SYMBOLS
        .chars()
        .position(|symbol| symbol == character)
        .map(|index| index as i64)
}

fn chunk_symbols(
    tokens: Vec<StyleTts2SymbolToken>,
    max_symbols_per_chunk: usize,
    chunking_enabled: bool,
) -> Vec<SynthesisChunk> {
    if tokens.is_empty() {
        return Vec::new();
    }
    if !chunking_enabled {
        return vec![chunk_from_tokens(tokens)];
    }

    split_required_question_chunks(tokens)
        .into_iter()
        .flat_map(|tokens| split_oversized_chunk(tokens, max_symbols_per_chunk))
        .into_iter()
        .filter(|chunk| !chunk.symbols.is_empty())
        .collect()
}

fn split_required_question_chunks(
    tokens: Vec<StyleTts2SymbolToken>,
) -> Vec<Vec<StyleTts2SymbolToken>> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    for token in tokens {
        let is_question_terminal =
            terminal_for_symbol(&token.symbol) == Some(TerminalPunctuation::Question);
        current.push(token);
        if is_question_terminal {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn split_oversized_chunk(
    mut tokens: Vec<StyleTts2SymbolToken>,
    max_symbols_per_chunk: usize,
) -> Vec<SynthesisChunk> {
    let mut chunks = Vec::new();
    while tokens.len() > max_symbols_per_chunk {
        let split_at = best_split_index(&tokens, max_symbols_per_chunk);
        let remainder = tokens.split_off(split_at);
        chunks.push(chunk_from_tokens(tokens));
        tokens = remainder;
    }
    if !tokens.is_empty() {
        chunks.push(chunk_from_tokens(tokens));
    }
    chunks
}

fn best_split_index(tokens: &[StyleTts2SymbolToken], max_symbols_per_chunk: usize) -> usize {
    let search_len = max_symbols_per_chunk.min(tokens.len());
    for index in (0..search_len).rev() {
        if terminal_for_symbol(&tokens[index].symbol).is_some() {
            return index + 1;
        }
    }
    for index in (0..search_len).rev() {
        if is_phrase_punctuation(&tokens[index]) {
            return index + 1;
        }
    }
    for index in (0..search_len).rev() {
        if is_word_boundary(&tokens[index]) {
            return index + 1;
        }
    }
    search_len.max(1)
}

fn chunk_from_tokens(tokens: Vec<StyleTts2SymbolToken>) -> SynthesisChunk {
    let terminal = tokens
        .last()
        .and_then(|token| terminal_for_symbol(&token.symbol));
    SynthesisChunk {
        symbols: tokens,
        terminal,
        source_text: None,
    }
}

fn is_phrase_punctuation(token: &StyleTts2SymbolToken) -> bool {
    token.source == StyleTts2SymbolSource::BoundaryPunctuation
        && matches!(token.symbol.as_str(), "," | ";" | ":")
}

fn is_word_boundary(token: &StyleTts2SymbolToken) -> bool {
    token.source == StyleTts2SymbolSource::Boundary || token.symbol == "|"
}

fn terminal_for_symbol(symbol: &str) -> Option<TerminalPunctuation> {
    match symbol {
        "." => Some(TerminalPunctuation::Period),
        "?" => Some(TerminalPunctuation::Question),
        "!" => Some(TerminalPunctuation::Exclamation),
        _ => None,
    }
}

fn invalid_output(reason: impl Into<String>) -> StyleTts2Error {
    StyleTts2Error::InvalidOutput {
        reason: reason.into(),
    }
}
