use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
#[cfg(feature = "piper-onnx")]
use ort::session::{builder::GraphOptimizationLevel, Session};
#[cfg(feature = "piper-onnx")]
use ort::value::{DynTensorValueType, Tensor, TensorElementType};
use serde_json::Value;
use speech::{
    FeatureId, FeatureValue, PauseKind, PhoneToken, PhonemeToken, ProsodicLabelKind, Spec,
    SpeechBoundaryToken, TerminalPunctuation, UtterancePlan,
};

#[derive(Debug, Clone, PartialEq)]
pub struct PiperVoiceConfig {
    pub sample_rate_hz: u32,
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
    pub num_speakers: Option<u32>,
    pub speaker_id_map: HashMap<String, u32>,
    pub length_scale: Option<f32>,
    pub noise_scale: Option<f32>,
    pub noise_w: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperPhonemeSequence {
    pub symbols: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperSynthesisChunk {
    pub sequence: PiperPhonemeSequence,
    pub pause_after_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperIdSequence {
    pub ids: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PiperSynthesisOutput {
    pub sample_rate_hz: u32,
    pub pcm_mono_f32: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PiperAudioChunk {
    pub chunk_index: usize,
    pub is_final: bool,
    pub pause_after_ms: u32,
    pub sample_rate_hz: u32,
    pub pcm_mono_f32: Vec<f32>,
}

pub trait PiperAudioSink {
    fn emit(&mut self, chunk: PiperAudioChunk) -> Result<()>;
}

impl<F> PiperAudioSink for F
where
    F: FnMut(PiperAudioChunk) -> Result<()>,
{
    fn emit(&mut self, chunk: PiperAudioChunk) -> Result<()> {
        self(chunk)
    }
}

#[cfg(feature = "piper-onnx")]
#[derive(Debug, Clone, PartialEq)]
struct PiperTensorSpec {
    name: String,
    tensor_type: Option<TensorElementType>,
}

#[cfg(feature = "piper-onnx")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PiperInferenceContract {
    id_input: String,
    id_lengths_input: String,
    scales_input: Option<String>,
    noise_scale_input: Option<String>,
    length_scale_input: Option<String>,
    noise_w_input: Option<String>,
    speaker_input: Option<String>,
    output_audio: String,
}

#[cfg(feature = "piper-onnx")]
#[derive(Debug)]
pub struct PiperOnnxBackend {
    config: PiperVoiceConfig,
    model_path: PathBuf,
    session: Session,
}

impl PiperVoiceConfig {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read Piper voice config {}", path.display()))?;
        Self::from_json_str(&json)
            .with_context(|| format!("failed to parse Piper voice config {}", path.display()))
    }

    pub fn from_json_str(json: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(json)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &Value) -> Result<Self> {
        let sample_rate_hz = parse_required_u32(
            value,
            &[&["audio", "sample_rate"], &["sample_rate"]],
            "audio.sample_rate",
        )?;
        let phoneme_id_map = parse_phoneme_id_map(
            find_value(value, &[&["phoneme_id_map"], &["phoneme_map"]])
                .context("missing required Piper voice config field `phoneme_id_map`")?,
        )?;
        let speaker_id_map = find_value(value, &[&["speaker_id_map"], &["speaker_map"]])
            .map(parse_speaker_id_map)
            .transpose()?
            .unwrap_or_default();
        let num_speakers = parse_optional_u32(
            value,
            &[&["num_speakers"], &["speaker_count"]],
            "num_speakers",
        )?
        .or_else(|| {
            if speaker_id_map.is_empty() {
                None
            } else {
                u32::try_from(speaker_id_map.len()).ok()
            }
        });

        Ok(Self {
            sample_rate_hz,
            phoneme_id_map,
            num_speakers,
            speaker_id_map,
            length_scale: parse_optional_f32(
                value,
                &[&["inference", "length_scale"], &["length_scale"]],
                "inference.length_scale",
            )?,
            noise_scale: parse_optional_f32(
                value,
                &[&["inference", "noise_scale"], &["noise_scale"]],
                "inference.noise_scale",
            )?,
            noise_w: parse_optional_f32(
                value,
                &[&["inference", "noise_w"], &["noise_w"]],
                "inference.noise_w",
            )?,
        })
    }
}

pub fn piper_voice_config_path(model_path: &Path) -> PathBuf {
    model_path.with_extension("onnx.json")
}

pub fn piper_sequence_from_plan(plan: &UtterancePlan) -> Result<PiperPhonemeSequence> {
    let mut symbols = Vec::new();
    if !plan.target_phones.is_empty() {
        let punctuation_after_words = typed_punctuation_after_words(&plan.boundaries)
            .or_else(|| plan.intended_text.as_deref().map(punctuation_after_words))
            .unwrap_or_default();
        let mut word_index = 0;
        let mut in_word = false;
        for (token_index, token) in plan.target_phones.iter().enumerate() {
            let Spec::Known(phone_id) = &token.phone else {
                continue;
            };
            if phone_id.0 == "boundary.word" {
                if in_word {
                    let boundary_symbol = punctuation_after_words
                        .get(word_index)
                        .and_then(|symbol| *symbol);
                    if boundary_symbol.is_some()
                        || !next_phone_is_epenthetic_linker(&plan.target_phones[token_index + 1..])
                    {
                        push_boundary_symbols(&mut symbols, boundary_symbol.unwrap_or(" "));
                    }
                    word_index += 1;
                    in_word = false;
                }
            } else if phone_id.0 == "boundary.letter" {
                continue;
            } else {
                let symbol = piper_symbol_for_phone(token).with_context(|| {
                    format!(
                        "cannot lower phone `{}` to a Piper ARPAbet symbol",
                        phone_id.0
                    )
                })?;
                push_symbol(&mut symbols, &symbol);
                in_word = true;
            }
        }
        if in_word {
            if let Some(symbol) = punctuation_after_words
                .get(word_index)
                .and_then(|symbol| *symbol)
            {
                push_symbol(&mut symbols, symbol);
            }
        }
    } else {
        for token in &plan.intended_phonemes {
            let Spec::Known(phoneme_id) = &token.phoneme else {
                continue;
            };
            let symbol = piper_symbol_for_phoneme(token).with_context(|| {
                format!(
                    "cannot lower phoneme `{}` to a Piper ARPAbet symbol",
                    phoneme_id.0
                )
            })?;
            push_symbol(&mut symbols, &symbol);
        }
    }
    apply_piper_prosody_terminal_hint(&mut symbols, plan);
    append_default_terminal_symbol(&mut symbols);
    Ok(PiperPhonemeSequence { symbols })
}

pub fn piper_synthesis_chunks_from_plan(plan: &UtterancePlan) -> Result<Vec<PiperSynthesisChunk>> {
    piper_sequence_from_plan(plan).map(piper_synthesis_chunks_from_sequence)
}

fn piper_synthesis_chunks_from_sequence(
    sequence: PiperPhonemeSequence,
) -> Vec<PiperSynthesisChunk> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut pending_pause_after_ms = None;
    let mut skip_leading_spaces = false;

    for symbol in sequence.symbols {
        if skip_leading_spaces && symbol == " " {
            continue;
        }
        skip_leading_spaces = false;

        let pause_after_ms = piper_pause_after_ms(&symbol);
        current.push(symbol);
        if let Some(pause_after_ms) = pause_after_ms {
            chunks.push(PiperSynthesisChunk {
                sequence: PiperPhonemeSequence {
                    symbols: std::mem::take(&mut current),
                },
                pause_after_ms: 0,
            });
            pending_pause_after_ms = Some(pause_after_ms);
            skip_leading_spaces = true;
        } else if let Some(pause_after_ms) = pending_pause_after_ms.take() {
            if let Some(previous) = chunks.last_mut() {
                previous.pause_after_ms = pause_after_ms;
            }
        }
    }

    if !current.is_empty() {
        chunks.push(PiperSynthesisChunk {
            sequence: PiperPhonemeSequence { symbols: current },
            pause_after_ms: 0,
        });
    }

    chunks
}

fn piper_pause_after_ms(symbol: &str) -> Option<u32> {
    match symbol {
        "," | ";" | ":" => Some(220),
        "." | "!" | "?" => Some(380),
        _ => None,
    }
}

fn piper_symbol_for_phone(token: &PhoneToken) -> Option<String> {
    let Spec::Known(phone_id) = &token.phone else {
        return None;
    };
    if let Some(symbol) = piper_arpabet_symbol_from_features(&token.features) {
        return Some(symbol);
    }
    piper_symbol_for_phone_id(phone_id.as_str()).map(str::to_string)
}

fn piper_symbol_for_phoneme(token: &PhonemeToken) -> Option<String> {
    piper_arpabet_symbol_from_features(&token.features)
}

fn piper_arpabet_symbol_from_features(features: &speech::FeatureBundle) -> Option<String> {
    let base = feature_category(features, "phonology.base_symbol")?;
    if !is_piper_arpabet_symbol(base) {
        return None;
    }
    if is_arpabet_vowel(base) {
        if let Some(stress) = feature_category(features, "phonology.stress").and_then(stress_digit)
        {
            return Some(format!("{base}{stress}"));
        }
    }
    Some(base.to_string())
}

fn piper_symbol_for_phone_id(phone_id: &str) -> Option<&'static str> {
    Some(match phone_id {
        "ipa.phone.ɑ" => "AA",
        "ipa.phone.æ" => "AE",
        "ipa.phone.ʌ" => "AH1",
        "ipa.phone.ə" | "ipa.phone.ɐ" => "AH0",
        "ipa.phone.ɔ" => "AO",
        "ipa.phone.aʊ" => "AW",
        "ipa.phone.aɪ" => "AY",
        "ipa.phone.b" => "B",
        "ipa.phone.tʃ" => "CH",
        "ipa.phone.d" => "D",
        "ipa.phone.ð" => "DH",
        "ipa.phone.ɛ" => "EH",
        "ipa.phone.ɝ" => "ER1",
        "ipa.phone.ɚ" => "ER0",
        "ipa.phone.eɪ" => "EY",
        "ipa.phone.f" => "F",
        "ipa.phone.ɡ" => "G",
        "ipa.phone.h" => "HH",
        "ipa.phone.ɪ" => "IH",
        "ipa.phone.iː" | "ipa.phone.i" => "IY",
        "ipa.phone.dʒ" => "JH",
        "ipa.phone.k" | "ipa.phone.kʰ" | "ipa.phone.k˭" => "K",
        "ipa.phone.l" | "ipa.phone.ɫ" => "L",
        "ipa.phone.m" => "M",
        "ipa.phone.n" => "N",
        "ipa.phone.ŋ" => "NG",
        "ipa.phone.oʊ" => "OW",
        "ipa.phone.ɔɪ" => "OY",
        "ipa.phone.p" | "ipa.phone.pʰ" | "ipa.phone.p˭" => "P",
        "ipa.phone.ɹ" => "R",
        "ipa.phone.s" => "S",
        "ipa.phone.ʃ" => "SH",
        "ipa.phone.t" | "ipa.phone.tʰ" | "ipa.phone.t˭" => "T",
        "ipa.phone.θ" => "TH",
        "ipa.phone.ʊ" => "UH",
        "ipa.phone.uː" | "ipa.phone.u" => "UW",
        "ipa.phone.v" => "V",
        "ipa.phone.w" => "W",
        "ipa.phone.j" => "Y",
        "ipa.phone.z" => "Z",
        "ipa.phone.ʒ" => "ZH",
        _ => return None,
    })
}

fn append_default_terminal_symbol(symbols: &mut Vec<String>) {
    if symbols.is_empty()
        || symbols
            .last()
            .is_some_and(|symbol| is_terminal_symbol(symbol))
    {
        return;
    }
    push_symbol(symbols, ".");
}

fn apply_piper_prosody_terminal_hint(symbols: &mut Vec<String>, plan: &UtterancePlan) {
    if !plan.target_prosody.labels.iter().any(|label| {
        matches!(
            label.kind,
            ProsodicLabelKind::QuestionRise | ProsodicLabelKind::AlternativeQuestionFall
        )
    }) {
        return;
    }

    if let Some(last) = symbols.last_mut() {
        if matches!(last.as_str(), "." | "!" | "?") {
            *last = "?".to_string();
            return;
        }
    }

    push_symbol(symbols, "?");
}

fn punctuation_after_words(text: &str) -> Vec<Option<&'static str>> {
    let word_spans = word_spans(text);
    word_spans
        .iter()
        .enumerate()
        .map(|(index, (_, end))| {
            let next_start = word_spans
                .get(index + 1)
                .map(|(start, _)| *start)
                .unwrap_or(text.len());
            punctuation_symbol(&text[*end..next_start])
        })
        .collect()
}

fn typed_punctuation_after_words(
    boundaries: &[SpeechBoundaryToken],
) -> Option<Vec<Option<&'static str>>> {
    let max_word_index = boundaries
        .iter()
        .filter(|boundary| typed_punctuation_symbol(boundary).is_some())
        .map(|boundary| boundary.after_grapheme_index)
        .max()?;
    let mut punctuation = vec![None; max_word_index + 1];
    for boundary in boundaries {
        if let Some(symbol) = typed_punctuation_symbol(boundary) {
            punctuation[boundary.after_grapheme_index] = Some(symbol);
        }
    }
    Some(punctuation)
}

fn typed_punctuation_symbol(boundary: &SpeechBoundaryToken) -> Option<&'static str> {
    if let Some(terminal) = boundary.terminal {
        return Some(match terminal {
            TerminalPunctuation::Period => ".",
            TerminalPunctuation::Question => "?",
            TerminalPunctuation::Exclamation => "!",
        });
    }
    if matches!(boundary.pause, Some(PauseKind::Comma)) {
        return Some(",");
    }
    None
}

fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (byte_index, character) in text.char_indices() {
        if is_word_chunk_character(character) {
            start.get_or_insert(byte_index);
            continue;
        }

        if let Some(start_byte) = start.take() {
            push_word_chunk_spans(text, start_byte, byte_index, &mut spans);
        }
    }

    if let Some(start_byte) = start {
        push_word_chunk_spans(text, start_byte, text.len(), &mut spans);
    }
    spans
}

fn is_word_chunk_character(character: char) -> bool {
    character.is_alphabetic() || is_apostrophe(character) || character == '-'
}

fn is_apostrophe(character: char) -> bool {
    matches!(character, '\'' | '’' | '‘' | 'ʼ')
}

fn push_word_chunk_spans(
    text: &str,
    start_byte: usize,
    end_byte: usize,
    spans: &mut Vec<(usize, usize)>,
) {
    let mut part_start = None;
    for (offset, character) in text[start_byte..end_byte].char_indices() {
        let byte_index = start_byte + offset;
        if character == '-' {
            if let Some(part_start_byte) = part_start.take() {
                push_camelcase_word_spans(text, part_start_byte, byte_index, spans);
            }
            continue;
        }

        part_start.get_or_insert(byte_index);
    }

    if let Some(part_start_byte) = part_start {
        push_camelcase_word_spans(text, part_start_byte, end_byte, spans);
    }
}

fn push_camelcase_word_spans(
    text: &str,
    start_byte: usize,
    end_byte: usize,
    spans: &mut Vec<(usize, usize)>,
) {
    let mut part_start = start_byte;
    let mut previous = None;
    let mut iterator = text[start_byte..end_byte].char_indices().peekable();
    while let Some((offset, character)) = iterator.next() {
        let byte_index = start_byte + offset;
        if let Some(previous_character) = previous {
            if should_split_camelcase_part(previous_character, character, iterator.peek()) {
                push_word_span(text, part_start, byte_index, spans);
                part_start = byte_index;
            }
        }
        previous = Some(character);
    }

    push_word_span(text, part_start, end_byte, spans);
}

fn should_split_camelcase_part(
    previous: char,
    current: char,
    next: Option<&(usize, char)>,
) -> bool {
    previous.is_lowercase()
        && current.is_uppercase()
        && next.is_some_and(|(_, next_char)| next_char.is_uppercase())
}

fn push_word_span(text: &str, start_byte: usize, end_byte: usize, spans: &mut Vec<(usize, usize)>) {
    let surface = &text[start_byte..end_byte];
    if surface
        .trim_matches(|character: char| !character.is_alphabetic())
        .is_empty()
    {
        return;
    }
    spans.push((start_byte, end_byte));
}

fn punctuation_symbol(text: &str) -> Option<&'static str> {
    text.chars().rev().find_map(|character| match character {
        '.' | '…' => Some("."),
        '!' => Some("!"),
        '?' => Some("?"),
        ',' => Some(","),
        ';' => Some(";"),
        ':' => Some(":"),
        _ => None,
    })
}

impl PiperPhonemeSequence {
    #[allow(dead_code)]
    pub fn to_symbols_compatible(&self, config: &PiperVoiceConfig) -> Result<Self> {
        let text_sequence = self.with_utterance_termination(config);
        validate_piper_plan_sequence(&text_sequence)?;
        if text_sequence
            .symbols
            .iter()
            .all(|symbol| config.phoneme_id_map.contains_key(symbol))
        {
            return Ok(text_sequence);
        }

        text_sequence.to_espeak_compatible(config)
    }

    pub fn to_text_ids_compatible(&self, config: &PiperVoiceConfig) -> Result<PiperIdSequence> {
        let text_sequence = self.with_utterance_termination(config);
        validate_piper_plan_sequence(&text_sequence)?;
        if config_has_piper_framing(config) {
            return text_sequence.to_framed_ids(config).or_else(|_| {
                text_sequence
                    .to_espeak_compatible(config)
                    .and_then(|sequence| sequence.to_framed_ids(config))
            });
        }

        text_sequence.to_ids(config).or_else(|_| {
            text_sequence
                .to_espeak_compatible(config)
                .and_then(|sequence| sequence.to_ids(config))
        })
    }

    fn to_ids(&self, config: &PiperVoiceConfig) -> Result<PiperIdSequence> {
        let mut ids = Vec::new();
        for symbol in &self.symbols {
            extend_symbol_ids(&mut ids, symbol, config)?;
        }
        Ok(PiperIdSequence { ids })
    }

    fn to_framed_ids(&self, config: &PiperVoiceConfig) -> Result<PiperIdSequence> {
        let mut ids = Vec::new();
        extend_symbol_ids(&mut ids, "^", config)?;
        extend_symbol_ids(&mut ids, "_", config)?;
        for symbol in &self.symbols {
            extend_symbol_ids(&mut ids, symbol, config)?;
            extend_symbol_ids(&mut ids, "_", config)?;
        }
        extend_symbol_ids(&mut ids, "$", config)?;
        Ok(PiperIdSequence { ids })
    }

    fn with_utterance_termination(&self, config: &PiperVoiceConfig) -> Self {
        if self.symbols.is_empty() {
            return self.clone();
        }

        let mut terminated = self.clone();
        if let Some(last) = terminated.symbols.last_mut() {
            if is_terminal_symbol(last) {
                if can_encode_piper_symbol(last, config) {
                    return terminated;
                }
                if let Some(symbol) = compatible_terminal_symbol(Some(last), config) {
                    *last = symbol.to_string();
                }
                return terminated;
            }
        }

        if let Some(symbol) = compatible_terminal_symbol(None, config) {
            terminated.symbols.push(symbol.to_string());
        }
        terminated
    }

    fn to_espeak_compatible(&self, config: &PiperVoiceConfig) -> Result<Self> {
        let mut symbols = Vec::new();
        for symbol in &self.symbols {
            let expanded = expand_espeak_phoneme(symbol, config)
                .with_context(|| format!("unknown Piper phoneme symbol `{symbol}`"))?;
            symbols.extend(expanded);
        }
        Ok(Self { symbols })
    }
}

#[cfg(feature = "piper-onnx")]
impl PiperOnnxBackend {
    pub fn load(model_path: impl AsRef<Path>, config: PiperVoiceConfig) -> Result<Self> {
        validate_config(&config)?;
        let model_path = model_path.as_ref().to_path_buf();
        ensure!(
            model_path.is_file(),
            "Piper ONNX model file not found at {}",
            model_path.display()
        );
        initialize_ort_runtime()?;

        let session = Session::builder()
            .context("failed to create Piper ONNX session builder")?
            .with_intra_threads(1)
            .map_err(|error| anyhow::anyhow!("failed to configure Piper ONNX threads: {error}"))?
            .with_inter_threads(1)
            .map_err(|error| anyhow::anyhow!("failed to configure Piper ONNX threads: {error}"))?
            .with_intra_op_spinning(false)
            .map_err(|error| anyhow::anyhow!("failed to configure Piper ONNX spinning: {error}"))?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|error| {
                anyhow::anyhow!("failed to configure Piper ONNX optimization: {error}")
            })?
            .commit_from_file(&model_path)
            .with_context(|| format!("failed to load Piper ONNX model {}", model_path.display()))?;

        Ok(Self {
            config,
            model_path,
            session,
        })
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.config.sample_rate_hz
    }

    #[allow(dead_code)]
    pub fn synthesize_plan(&mut self, plan: &UtterancePlan) -> Result<PiperSynthesisOutput> {
        let mut pcm_mono_f32 = Vec::new();
        self.synthesize_plan_streaming(plan, &mut |chunk: PiperAudioChunk| {
            pcm_mono_f32.extend(chunk.pcm_mono_f32);
            Ok(())
        })?;

        Ok(PiperSynthesisOutput {
            sample_rate_hz: self.config.sample_rate_hz,
            pcm_mono_f32,
        })
    }

    pub fn synthesize_plan_streaming(
        &mut self,
        plan: &UtterancePlan,
        sink: &mut dyn PiperAudioSink,
    ) -> Result<()> {
        let chunks = piper_synthesis_chunks_from_plan(plan)?;
        let chunk_count = chunks.len();
        for (chunk_index, chunk) in chunks.into_iter().enumerate() {
            let ids = chunk
                .sequence
                .to_text_ids_compatible(&self.config)
                .context("failed to map Mortar speech plan to Piper phoneme IDs")?;
            let mut output = self.synthesize_ids(&ids)?.pcm_mono_f32;
            output.extend(std::iter::repeat(0.0).take(pause_sample_count(
                self.config.sample_rate_hz,
                chunk.pause_after_ms,
            )));
            sink.emit(PiperAudioChunk {
                chunk_index,
                is_final: chunk_index + 1 == chunk_count,
                pause_after_ms: chunk.pause_after_ms,
                sample_rate_hz: self.config.sample_rate_hz,
                pcm_mono_f32: output,
            })?;
        }

        Ok(())
    }

    pub fn synthesize_ids(&mut self, ids: &PiperIdSequence) -> Result<PiperSynthesisOutput> {
        ensure!(
            !ids.ids.is_empty(),
            "Piper ID sequence cannot be empty for ONNX synthesis"
        );

        let input_specs = self
            .session
            .inputs()
            .iter()
            .map(|input| PiperTensorSpec {
                name: input.name().to_string(),
                tensor_type: input.dtype().tensor_type(),
            })
            .collect::<Vec<_>>();
        let output_specs = self
            .session
            .outputs()
            .iter()
            .map(|output| PiperTensorSpec {
                name: output.name().to_string(),
                tensor_type: output.dtype().tensor_type(),
            })
            .collect::<Vec<_>>();
        let contract = resolve_inference_contract(
            &input_specs,
            &output_specs,
            &self.config,
            &self.model_path,
        )?;
        let ids_len = i64::try_from(ids.ids.len()).context("Piper ID sequence is too long")?;
        let scales = inference_scales(&self.config);
        let mut inputs = Vec::with_capacity(6);

        inputs.push((
            contract.id_input.clone(),
            Tensor::from_array((vec![1_i64, ids_len], ids.ids.clone()))
                .context("failed to build Piper ONNX ID tensor")?
                .upcast(),
        ));
        inputs.push((
            contract.id_lengths_input.clone(),
            Tensor::from_array((vec![1_i64], vec![ids_len]))
                .context("failed to build Piper ONNX length tensor")?
                .upcast(),
        ));
        if let Some(name) = &contract.scales_input {
            inputs.push((
                name.clone(),
                Tensor::from_array((vec![3_i64], scales.to_vec()))
                    .with_context(|| format!("failed to build Piper ONNX `{name}` tensor"))?
                    .upcast(),
            ));
        }
        if let Some(name) = &contract.noise_scale_input {
            inputs.push((
                name.clone(),
                Tensor::from_array((vec![1_i64], vec![scales[0]]))
                    .with_context(|| format!("failed to build Piper ONNX `{name}` tensor"))?
                    .upcast(),
            ));
        }
        if let Some(name) = &contract.length_scale_input {
            inputs.push((
                name.clone(),
                Tensor::from_array((vec![1_i64], vec![scales[1]]))
                    .with_context(|| format!("failed to build Piper ONNX `{name}` tensor"))?
                    .upcast(),
            ));
        }
        if let Some(name) = &contract.noise_w_input {
            inputs.push((
                name.clone(),
                Tensor::from_array((vec![1_i64], vec![scales[2]]))
                    .with_context(|| format!("failed to build Piper ONNX `{name}` tensor"))?
                    .upcast(),
            ));
        }
        if let Some(name) = &contract.speaker_input {
            inputs.push((
                name.clone(),
                Tensor::from_array((vec![1_i64], vec![0_i64]))
                    .with_context(|| format!("failed to build Piper ONNX `{name}` tensor"))?
                    .upcast(),
            ));
        }

        let outputs = self.session.run(inputs).with_context(|| {
            format!(
                "failed to run Piper ONNX inference for model {}",
                self.model_path.display()
            )
        })?;
        let output = outputs
            .get(contract.output_audio.as_str())
            .with_context(|| {
                format!(
                    "Piper ONNX inference did not return `{}`",
                    contract.output_audio
                )
            })?;
        let output = output
            .downcast_ref::<DynTensorValueType>()
            .with_context(|| {
                format!(
                    "Piper ONNX output `{}` is not a tensor",
                    contract.output_audio
                )
            })?;
        let (_, samples) = output
            .try_extract_tensor::<f32>()
            .with_context(|| format!("Piper ONNX output `{}` is not f32", contract.output_audio))?;
        ensure!(
            !samples.is_empty(),
            "Piper ONNX inference returned an empty waveform output"
        );

        Ok(PiperSynthesisOutput {
            sample_rate_hz: self.config.sample_rate_hz,
            pcm_mono_f32: samples.to_vec(),
        })
    }
}

#[cfg(not(feature = "piper-onnx"))]
pub struct PiperOnnxBackend;

#[cfg(not(feature = "piper-onnx"))]
impl PiperOnnxBackend {
    pub fn load(_model_path: impl AsRef<Path>, _config: PiperVoiceConfig) -> Result<Self> {
        bail!("Piper ONNX synthesis requires building with the `piper-onnx` feature")
    }

    pub fn sample_rate_hz(&self) -> u32 {
        0
    }

    #[allow(dead_code)]
    pub fn synthesize_plan(&mut self, _plan: &UtterancePlan) -> Result<PiperSynthesisOutput> {
        bail!("Piper ONNX synthesis requires building with the `piper-onnx` feature")
    }

    pub fn synthesize_plan_streaming(
        &mut self,
        _plan: &UtterancePlan,
        _sink: &mut dyn PiperAudioSink,
    ) -> Result<()> {
        bail!("Piper ONNX synthesis requires building with the `piper-onnx` feature")
    }
}

fn push_symbol(symbols: &mut Vec<String>, symbol: &str) {
    if symbol == " " && symbols.last().is_some_and(|last| last == " ") {
        return;
    }
    symbols.push(symbol.to_string());
}

#[cfg(any(feature = "piper-onnx", test))]
fn pause_sample_count(sample_rate_hz: u32, pause_ms: u32) -> usize {
    ((sample_rate_hz as u128 * pause_ms as u128) / 1000) as usize
}

fn push_boundary_symbols(symbols: &mut Vec<String>, symbol: &str) {
    push_symbol(symbols, symbol);
    if is_clause_pause_symbol(symbol) {
        push_symbol(symbols, " ");
    }
}

fn is_clause_pause_symbol(symbol: &str) -> bool {
    matches!(symbol, "," | ";" | ":")
}

fn next_phone_is_epenthetic_linker(tokens: &[PhoneToken]) -> bool {
    for token in tokens {
        if let Spec::Known(id) = &token.phone {
            if id.as_str().starts_with("boundary.") {
                continue;
            }
        }
        return is_epenthetic_phone(token);
    }
    false
}

fn is_epenthetic_phone(token: &PhoneToken) -> bool {
    token.provenance.method.contains("epenthesis rule")
}

fn find_value<'a>(root: &'a Value, paths: &[&[&str]]) -> Option<&'a Value> {
    paths.iter().find_map(|path| {
        let mut current = root;
        for segment in *path {
            current = current.get(*segment)?;
        }
        Some(current)
    })
}

fn parse_required_u32(root: &Value, paths: &[&[&str]], field: &'static str) -> Result<u32> {
    let value = find_value(root, paths)
        .with_context(|| format!("missing required Piper voice config field `{field}`"))?;
    parse_u32(value, field)
}

fn parse_optional_u32(root: &Value, paths: &[&[&str]], field: &'static str) -> Result<Option<u32>> {
    find_value(root, paths)
        .map(|value| parse_u32(value, field))
        .transpose()
}

fn parse_u32(value: &Value, field: &'static str) -> Result<u32> {
    let number = value
        .as_u64()
        .with_context(|| format!("invalid Piper voice config field `{field}`: expected integer"))?;
    Ok(u32::try_from(number)
        .with_context(|| format!("invalid Piper voice config field `{field}`: exceeds u32"))?)
}

fn parse_optional_f32(root: &Value, paths: &[&[&str]], field: &'static str) -> Result<Option<f32>> {
    find_value(root, paths)
        .map(|value| parse_f32(value, field))
        .transpose()
}

fn parse_f32(value: &Value, field: &'static str) -> Result<f32> {
    let number = value
        .as_f64()
        .with_context(|| format!("invalid Piper voice config field `{field}`: expected number"))?;
    ensure!(
        number.is_finite() && number >= f32::MIN as f64 && number <= f32::MAX as f64,
        "invalid Piper voice config field `{field}`: value is out of f32 range"
    );
    Ok(number as f32)
}

fn parse_phoneme_id_map(value: &Value) -> Result<HashMap<String, Vec<i64>>> {
    let entries = value
        .as_object()
        .context("invalid Piper voice config field `phoneme_id_map`: expected object")?;
    let mut map = HashMap::with_capacity(entries.len());
    for (symbol, ids) in entries {
        let ids = match ids {
            Value::Array(values) => values.iter().map(parse_i64).collect::<Result<Vec<_>>>()?,
            _ => vec![parse_i64(ids)?],
        };
        map.insert(symbol.clone(), ids);
    }
    Ok(map)
}

fn parse_speaker_id_map(value: &Value) -> Result<HashMap<String, u32>> {
    let entries = value
        .as_object()
        .context("invalid Piper voice config field `speaker_id_map`: expected object")?;
    let mut map = HashMap::with_capacity(entries.len());
    for (speaker, id) in entries {
        map.insert(speaker.clone(), parse_u32(id, "speaker_id_map")?);
    }
    Ok(map)
}

fn parse_i64(value: &Value) -> Result<i64> {
    value
        .as_i64()
        .context("invalid Piper voice config field `phoneme_id_map`: expected integer")
}

fn extend_symbol_ids(ids: &mut Vec<i64>, symbol: &str, config: &PiperVoiceConfig) -> Result<()> {
    let mapped = config
        .phoneme_id_map
        .get(symbol)
        .with_context(|| format!("unknown Piper phoneme symbol `{symbol}`"))?;
    ids.extend(mapped);
    Ok(())
}

fn config_has_piper_framing(config: &PiperVoiceConfig) -> bool {
    ["^", "_", "$"]
        .iter()
        .all(|symbol| config.phoneme_id_map.contains_key(*symbol))
}

fn is_terminal_symbol(symbol: &str) -> bool {
    matches!(symbol, "|" | "." | "!" | "?" | "$")
}

fn can_encode_piper_symbol(symbol: &str, config: &PiperVoiceConfig) -> bool {
    config.phoneme_id_map.contains_key(symbol) || expand_espeak_phoneme(symbol, config).is_some()
}

fn validate_piper_plan_sequence(sequence: &PiperPhonemeSequence) -> Result<()> {
    for symbol in &sequence.symbols {
        ensure!(
            is_piper_plan_symbol(symbol),
            "unsupported pre-compat Piper symbol `{symbol}`; expected ARPAbet, space, or punctuation"
        );
    }
    Ok(())
}

fn is_piper_plan_symbol(symbol: &str) -> bool {
    if matches!(symbol, " " | "|" | "." | "!" | "?" | "," | ";" | ":") {
        return true;
    }
    let (base, stress) = split_arpabet_stress(symbol);
    if stress.is_some() && !is_arpabet_vowel(base) {
        return false;
    }
    is_piper_arpabet_symbol(base)
}

fn split_arpabet_stress(symbol: &str) -> (&str, Option<char>) {
    match symbol.chars().last() {
        Some(stress @ ('0' | '1' | '2')) => (&symbol[..symbol.len() - 1], Some(stress)),
        _ => (symbol, None),
    }
}

fn compatible_terminal_symbol<'a>(
    requested: Option<&'a str>,
    config: &PiperVoiceConfig,
) -> Option<&'a str> {
    if let Some(symbol) = requested {
        if can_encode_piper_symbol(symbol, config) {
            return Some(symbol);
        }
    }
    if can_encode_piper_symbol(".", config) {
        return Some(".");
    }
    if can_encode_piper_symbol("|", config) {
        return Some("|");
    }
    None
}

fn expand_espeak_phoneme(symbol: &str, config: &PiperVoiceConfig) -> Option<Vec<String>> {
    if symbol == " " {
        return if config.phoneme_id_map.contains_key(" ") {
            Some(vec![" ".to_string()])
        } else {
            Some(Vec::new())
        };
    }

    let stress_marker = match symbol.chars().last() {
        Some('1') => Some("ˈ"),
        Some('2') => Some("ˌ"),
        _ => None,
    };
    let base_symbol = symbol
        .strip_suffix(['0', '1', '2'])
        .filter(|base| is_arpabet_vowel(base))
        .unwrap_or(symbol);

    if base_symbol != symbol && config.phoneme_id_map.contains_key(base_symbol) {
        let mut output = Vec::new();
        if let Some(marker) = stress_marker {
            if config.phoneme_id_map.contains_key(marker) {
                output.push(marker.to_string());
            }
        }
        output.push(base_symbol.to_string());
        return Some(output);
    }

    if config.phoneme_id_map.contains_key(symbol) {
        return Some(vec![symbol.to_string()]);
    }

    let expanded = match (symbol, base_symbol) {
        ("AH0", _) => &["ə"][..],
        ("AH1" | "AH2", _) => &["ʌ"],
        (_, "AA") => &["ɑ"],
        (_, "AH") => &["ə"],
        (_, "AY") => &["a", "ɪ"],
        (_, "AE") => &["æ"],
        (_, "AO") => &["ɔ"],
        (_, "AW") => &["a", "ʊ"],
        (_, "B") => &["b"],
        (_, "CH") => &["t", "ʃ"],
        (_, "D") => &["d"],
        (_, "DH") => &["ð"],
        (_, "DX") => &["ɾ"],
        (_, "EH") => &["ɛ"],
        (_, "ER") => &["ɚ"],
        (_, "EY") => &["e", "ɪ"],
        (_, "F") => &["f"],
        (_, "G") => &["ɡ"],
        (_, "HH") => &["h"],
        (_, "IH") => &["ɪ"],
        (_, "IY") => &["i"],
        (_, "JH") => &["d", "ʒ"],
        (_, "K") => &["k"],
        (_, "L") => &["l"],
        (_, "M") => &["m"],
        (_, "N") => &["n"],
        (_, "NG") => &["ŋ"],
        (_, "OW") => &["o", "ʊ"],
        (_, "OY") => &["ɔ", "ɪ"],
        (_, "P") => &["p"],
        (_, "R") => &["ɹ"],
        (_, "S") => &["s"],
        (_, "SH") => &["ʃ"],
        (_, "T") => &["t"],
        (_, "TH") => &["θ"],
        (_, "TS") => &["t", "s"],
        (_, "UH") => &["ʊ"],
        (_, "UW") => &["u"],
        (_, "V") => &["v"],
        (_, "W") => &["w"],
        (_, "Y") => &["j"],
        (_, "Z") => &["z"],
        (_, "ZH") => &["ʒ"],
        (_, "|") => &["."],
        _ => return None,
    };

    let mut expanded = expanded
        .iter()
        .map(|sym| (*sym).to_string())
        .collect::<Vec<_>>();
    if config.phoneme_id_map.contains_key("ː") && matches!(base_symbol, "AA" | "AO" | "IY" | "UW")
    {
        expanded.push("ː".to_string());
    }
    if !expanded
        .iter()
        .all(|sym| config.phoneme_id_map.contains_key(sym))
    {
        return None;
    }

    let mut output = Vec::new();
    if let Some(marker) = stress_marker {
        if config.phoneme_id_map.contains_key(marker) {
            output.push(marker.to_string());
        }
    }
    output.extend(expanded);
    Some(output)
}

fn feature_category<'a>(features: &'a speech::FeatureBundle, feature_id: &str) -> Option<&'a str> {
    let value = features.values.get(&FeatureId(feature_id.into()))?;
    match value {
        Spec::Known(FeatureValue::Category(val)) | Spec::Known(FeatureValue::Text(val)) => {
            Some(val.as_str())
        }
        _ => None,
    }
}

fn stress_digit(stress: &str) -> Option<&'static str> {
    match stress {
        "unstressed" => Some("0"),
        "primary" => Some("1"),
        "secondary" => Some("2"),
        _ => None,
    }
}

fn is_piper_arpabet_symbol(symbol: &str) -> bool {
    is_arpabet_vowel(symbol)
        || matches!(
            symbol,
            "B" | "CH"
                | "D"
                | "DH"
                | "DX"
                | "F"
                | "G"
                | "HH"
                | "JH"
                | "K"
                | "L"
                | "M"
                | "N"
                | "NG"
                | "P"
                | "R"
                | "S"
                | "SH"
                | "T"
                | "TH"
                | "TS"
                | "V"
                | "W"
                | "Y"
                | "Z"
                | "ZH"
        )
}

fn is_arpabet_vowel(symbol: &str) -> bool {
    matches!(
        symbol,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
}

#[cfg(feature = "piper-onnx")]
fn validate_config(config: &PiperVoiceConfig) -> Result<()> {
    ensure!(
        config.sample_rate_hz > 0,
        "missing required Piper voice config field `audio.sample_rate`"
    );
    ensure!(
        !config.phoneme_id_map.is_empty(),
        "missing required Piper voice config field `phoneme_id_map`"
    );
    if let Some(num_speakers) = config.num_speakers {
        ensure!(
            num_speakers > 0,
            "invalid Piper voice config field `num_speakers`: expected a value greater than zero"
        );
    }
    Ok(())
}

#[cfg(feature = "piper-onnx")]
fn resolve_inference_contract(
    input_specs: &[PiperTensorSpec],
    output_specs: &[PiperTensorSpec],
    config: &PiperVoiceConfig,
    model_path: &Path,
) -> Result<PiperInferenceContract> {
    ensure!(
        !input_specs.is_empty(),
        "Piper ONNX model `{}` exposes no inputs",
        model_path.display()
    );
    ensure!(
        !output_specs.is_empty(),
        "Piper ONNX model `{}` exposes no outputs",
        model_path.display()
    );

    let id_input = resolve_required_tensor_input(
        input_specs,
        &["input", "input_ids", "phoneme_ids", "ids"],
        TensorElementType::Int64,
        "phoneme ID input tensor",
        model_path,
    )?;
    let id_lengths_input = resolve_required_tensor_input(
        input_specs,
        &["input_lengths", "lengths", "input_lengths_tensor"],
        TensorElementType::Int64,
        "phoneme length input tensor",
        model_path,
    )?;
    let scales_input =
        resolve_optional_tensor_input(input_specs, &["scales"], TensorElementType::Float32)?;
    let noise_scale_input =
        resolve_optional_tensor_input(input_specs, &["noise_scale"], TensorElementType::Float32)?;
    let length_scale_input =
        resolve_optional_tensor_input(input_specs, &["length_scale"], TensorElementType::Float32)?;
    let noise_w_input =
        resolve_optional_tensor_input(input_specs, &["noise_w"], TensorElementType::Float32)?;
    let speaker_input = resolve_optional_tensor_input(
        input_specs,
        &["sid", "speaker_id"],
        TensorElementType::Int64,
    )?;

    let speaker_count = match config.num_speakers {
        Some(num_speakers) => num_speakers,
        None => u32::try_from(config.speaker_id_map.len()).context("invalid Piper speaker map")?,
    };
    if speaker_count > 1 {
        bail!(
            "Piper ONNX multi-speaker inference is not supported yet for `{}`",
            model_path.display()
        );
    }

    let supported = [
        Some(id_input.clone()),
        Some(id_lengths_input.clone()),
        scales_input.clone(),
        noise_scale_input.clone(),
        length_scale_input.clone(),
        noise_w_input.clone(),
        speaker_input.clone(),
    ];
    for input in input_specs {
        if !supported.iter().flatten().any(|name| name == &input.name) {
            bail!(
                "unsupported Piper ONNX input `{}` for model `{}`",
                input.name,
                model_path.display()
            );
        }
    }

    let output_audio = resolve_required_tensor_output(
        output_specs,
        &["output", "audio", "waveform"],
        TensorElementType::Float32,
        "audio output tensor",
        model_path,
    )?;
    if output_specs.iter().any(|spec| {
        spec.name != output_audio && spec.tensor_type == Some(TensorElementType::Float32)
    }) {
        bail!(
            "unsupported Piper ONNX model `{}` contract: multiple f32 outputs detected",
            model_path.display()
        );
    }

    Ok(PiperInferenceContract {
        id_input,
        id_lengths_input,
        scales_input,
        noise_scale_input,
        length_scale_input,
        noise_w_input,
        speaker_input,
        output_audio,
    })
}

#[cfg(feature = "piper-onnx")]
fn resolve_required_tensor_input(
    inputs: &[PiperTensorSpec],
    aliases: &[&str],
    expected_type: TensorElementType,
    label: &str,
    model_path: &Path,
) -> Result<String> {
    let input = resolve_tensor_by_alias(inputs, aliases).with_context(|| {
        format!(
            "unsupported Piper ONNX model contract for `{}`: missing {}",
            model_path.display(),
            label
        )
    })?;
    ensure!(
        input.tensor_type == Some(expected_type),
        "unsupported Piper ONNX model contract for `{}`: input `{}` expected type {:?}, got {:?}",
        model_path.display(),
        input.name,
        expected_type,
        input.tensor_type
    );
    Ok(input.name.clone())
}

#[cfg(feature = "piper-onnx")]
fn resolve_optional_tensor_input(
    inputs: &[PiperTensorSpec],
    aliases: &[&str],
    expected_type: TensorElementType,
) -> Result<Option<String>> {
    let Some(input) = resolve_tensor_by_alias(inputs, aliases) else {
        return Ok(None);
    };
    ensure!(
        input.tensor_type == Some(expected_type),
        "unsupported Piper ONNX model contract: input `{}` expected type {:?}, got {:?}",
        input.name,
        expected_type,
        input.tensor_type
    );
    Ok(Some(input.name.clone()))
}

#[cfg(feature = "piper-onnx")]
fn resolve_required_tensor_output(
    outputs: &[PiperTensorSpec],
    aliases: &[&str],
    expected_type: TensorElementType,
    label: &str,
    model_path: &Path,
) -> Result<String> {
    let output = resolve_tensor_by_alias(outputs, aliases)
        .or_else(|| {
            outputs
                .iter()
                .find(|spec| spec.tensor_type == Some(expected_type))
        })
        .with_context(|| {
            format!(
                "unsupported Piper ONNX model contract for `{}`: missing {}",
                model_path.display(),
                label
            )
        })?;
    ensure!(
        output.tensor_type == Some(expected_type),
        "unsupported Piper ONNX model contract for `{}`: output `{}` expected type {:?}, got {:?}",
        model_path.display(),
        output.name,
        expected_type,
        output.tensor_type
    );
    Ok(output.name.clone())
}

#[cfg(feature = "piper-onnx")]
fn resolve_tensor_by_alias<'a>(
    specs: &'a [PiperTensorSpec],
    aliases: &[&str],
) -> Option<&'a PiperTensorSpec> {
    aliases
        .iter()
        .find_map(|alias| specs.iter().find(|spec| spec.name == *alias))
}

#[cfg(feature = "piper-onnx")]
fn inference_scales(config: &PiperVoiceConfig) -> [f32; 3] {
    [
        config.noise_scale.unwrap_or(0.667),
        config.length_scale.unwrap_or(1.0),
        config.noise_w.unwrap_or(0.8),
    ]
}

#[cfg(feature = "piper-onnx")]
fn initialize_ort_runtime() -> Result<()> {
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH").filter(|value| !value.is_empty()) {
        let path = PathBuf::from(path);
        ensure!(
            path.is_file(),
            "ORT_DYLIB_PATH points to {}, but that file does not exist",
            path.display()
        );
        ort::init_from(&path)
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to load ONNX Runtime dynamic library from {}: {error}",
                    path.display()
                )
            })?
            .commit();
        return Ok(());
    }

    if let Some(path) = find_onnxruntime_dylib() {
        ort::init_from(&path)
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to load ONNX Runtime dynamic library from {}: {error}",
                    path.display()
                )
            })?
            .commit();
        Ok(())
    } else {
        bail!(
            "Piper ONNX requires an ONNX Runtime shared library. Install ONNX Runtime or set ORT_DYLIB_PATH to libonnxruntime.so."
        )
    }
}

#[cfg(feature = "piper-onnx")]
fn find_onnxruntime_dylib() -> Option<PathBuf> {
    let mut search_dirs = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        if let Ok(entries) = std::fs::read_dir(home.join(".local/lib")) {
            search_dirs.extend(entries.flatten().filter_map(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("python")
                    .then(|| entry.path().join("site-packages/onnxruntime/capi"))
            }));
        }
        for extensions_dir in [
            home.join(".vscode/extensions"),
            home.join(".vscode-server/extensions"),
        ] {
            if let Ok(entries) = std::fs::read_dir(extensions_dir) {
                search_dirs.extend(
                    entries
                        .flatten()
                        .filter_map(|entry| {
                            let file_name = entry.file_name();
                            if file_name.to_string_lossy().contains("windows-ai-studio") {
                                Some(vec![
                                    entry.path().join("bin"),
                                    entry.path().join("ai-mlstudio/bin"),
                                    entry.path().join("ai-foundry/bin"),
                                ])
                            } else {
                                None
                            }
                        })
                        .flatten(),
                );
            }
        }
    }
    if let Some(paths) = std::env::var_os("LD_LIBRARY_PATH") {
        search_dirs.extend(std::env::split_paths(&paths));
    }
    search_dirs.extend([
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/local/lib64"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/lib/x86_64-linux-gnu"),
    ]);

    let mut candidates = Vec::new();
    for dir in search_dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        candidates.extend(entries.flatten().filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            (name == "libonnxruntime.so" || name.starts_with("libonnxruntime.so."))
                .then(|| entry.path())
        }));
    }
    candidates.sort();
    candidates.pop()
}
