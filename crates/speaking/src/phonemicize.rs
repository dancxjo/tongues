use std::fmt;
use std::sync::{OnceLock, RwLock};

use serde::{Deserialize, Serialize};

use crate::data::lexicons::cmudict::CmuStress;
use crate::data::lexicons::{self, CMUDICT_ID, LEXIQUE383_ID, PronunciationStatus};
use crate::data::notation::{self, PronunciationNotation};
use crate::data::varieties::PRONUNCIATION_PIPELINE_VARIETY_DATA;
use crate::data::{canonical_variety_id, variety_by_code};
use crate::evidence::{EvidenceProvenance, EvidenceSource};
use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, GraphemeId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::GraphemeToken;
use crate::phonology::{PhoneToken, PhonemeToken};
use crate::prosody::{ProsodicLabel, ProsodicLabelKind, ProsodyTrack, Syllable};
use crate::realize::{
    PhoneDecompositionPolicy, RealizationOptions, epenthetic_phones_after, realize_phoneme_at,
    realize_phonemes,
};
use crate::segment::{BoundaryKind, PauseKind, SpeechBoundaryToken, TerminalPunctuation};
use crate::spec::Spec;
use crate::syllabify::syllabify_phones;
use crate::syntax::{GrammarParser, PartOfSpeech, SentenceSyntaxAnalysis, VarietyGrammarParser};
use crate::time::{TextSpan, TimeSpan};
use crate::variety::{
    ConnectedSpeechRule, LinguisticVariety, NumberNormalizationProfile, OrthographicUnitKind,
    WeakFormFollowingContext, WeakFormRule, WeakFormStyleContext,
};

const WORD_BOUNDARY_ID: &str = "boundary.word";
const LETTER_BOUNDARY_ID: &str = "boundary.letter";
const NO_LETTER_INDEX: usize = usize::MAX;

type PhonemicizerFactory = fn() -> Box<dyn Phonemicizer>;
pub type UnknownPronunciationFallback = fn(word: &str, variety: &str) -> Option<String>;

static UNKNOWN_PRONUNCIATION_FALLBACK: OnceLock<RwLock<Option<UnknownPronunciationFallback>>> =
    OnceLock::new();

pub fn set_unknown_pronunciation_fallback(fallback: Option<UnknownPronunciationFallback>) {
    let lock = UNKNOWN_PRONUNCIATION_FALLBACK.get_or_init(|| RwLock::new(None));
    if let Ok(mut configured) = lock.write() {
        *configured = fallback;
    }
}

fn unknown_pronunciation_fallback() -> Option<UnknownPronunciationFallback> {
    UNKNOWN_PRONUNCIATION_FALLBACK
        .get()
        .and_then(|lock| lock.read().ok().and_then(|configured| *configured))
}

struct PronunciationPipelineRegistration {
    id: &'static str,
    phonemicizer: PhonemicizerFactory,
}

const PRONUNCIATION_PIPELINE_REGISTRY: &[PronunciationPipelineRegistration] =
    &[PronunciationPipelineRegistration {
        id: PRONUNCIATION_PIPELINE_VARIETY_DATA,
        phonemicizer: variety_data_phonemicizer,
    }];

fn variety_data_phonemicizer() -> Box<dyn Phonemicizer> {
    Box::new(VarietyDataPhonemicizer)
}

pub trait Phonemicizer {
    fn phonemicize(
        &self,
        input: &PhonemicizeRequest,
    ) -> Result<PhonemicizeOutput, PhonemicizeError>;
}

pub fn phonemicizer_for_variety(
    variety: &VarietyId,
) -> Result<Box<dyn Phonemicizer>, PhonemicizeError> {
    let canonical =
        canonical_variety_id(&variety.0).ok_or_else(|| PhonemicizeError::UnsupportedVariety {
            variety: variety.clone(),
        })?;
    let variety_data =
        variety_by_code(&canonical.0).ok_or_else(|| PhonemicizeError::UnsupportedVariety {
            variety: canonical.clone(),
        })?;
    let pipeline_id = variety_data
        .pronunciation_pipeline
        .as_deref()
        .unwrap_or(PRONUNCIATION_PIPELINE_VARIETY_DATA);
    PRONUNCIATION_PIPELINE_REGISTRY
        .iter()
        .find(|registration| registration.id == pipeline_id)
        .map(|registration| (registration.phonemicizer)())
        .ok_or(PhonemicizeError::UnsupportedVariety { variety: canonical })
}

pub trait PronunciationPipeline {
    fn canonical_variety_id(
        &self,
        requested_variety: &VarietyId,
    ) -> Result<VarietyId, PhonemicizeError>;

    fn variety(&self, canonical_variety: &VarietyId)
    -> Result<LinguisticVariety, PhonemicizeError>;

    fn text_normalizer(&self, text: &str, variety: &VarietyId) -> String {
        normalize_text_for_variety(text, variety)
    }

    fn normalize_numbers(&self, text: &str, _variety: &VarietyId) -> String {
        text.to_string()
    }

    fn orthographic_tokenizer(&self, text: &str) -> Vec<WordToken>;

    fn boundary_extractor(
        &self,
        text: &str,
        words: &[WordToken],
        variety: &LinguisticVariety,
    ) -> Vec<SpeechBoundaryToken>;

    fn weak_form_resolver(
        &self,
        word: &WordToken,
        variety: &LinguisticVariety,
        context: TokenPronunciationContext,
    ) -> Option<WordPronunciation>;

    fn token_classifier(
        &self,
        word: &WordToken,
        variety: &LinguisticVariety,
        context: TokenPronunciationContext,
    ) -> WordPronunciation;

    fn uses_generic_orthography_before_unknown(&self) -> bool {
        true
    }

    fn unknown_word_pronunciation(
        &self,
        word: &WordToken,
        variety: &LinguisticVariety,
        context: TokenPronunciationContext,
    ) -> WordPronunciation {
        if let Some(pronunciation) = fallback_pronunciation(word, variety, context) {
            return pronunciation;
        }
        missing_pronunciation(word, context, "missing variety-data pronunciation")
    }

    fn syntax_analysis(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
        variety: &LinguisticVariety,
    ) -> Option<SentenceSyntaxAnalysis> {
        Some(VarietyGrammarParser::new(variety.id.clone()).parse(words, terminal))
    }

    fn annotate_boundaries(
        &self,
        boundaries: &mut Vec<SpeechBoundaryToken>,
        words: &[WordToken],
        syntax: &SentenceSyntaxAnalysis,
        variety: &LinguisticVariety,
    ) {
        annotate_alternative_question_boundaries(boundaries, words, syntax, variety);
    }

    fn prosodic_label_for_boundary(
        &self,
        boundary: &SpeechBoundaryToken,
        words: &[WordToken],
        sentence_start_word_index: usize,
        variety: &LinguisticVariety,
    ) -> Option<ProsodicLabelKind> {
        match (boundary.terminal, boundary.pause) {
            (Some(TerminalPunctuation::Question), _) => {
                let sentence_words = words_in_sentence(words, sentence_start_word_index, boundary);
                Some(prosodic_label_for_question(sentence_words, variety))
            }
            _ => generic_prosodic_label_for_boundary(boundary),
        }
    }

    fn phoneme_planner(
        &self,
        variety_id: &VarietyId,
        word_index: usize,
        pronunciation: &WordPronunciation,
    ) -> Vec<PhonemeToken> {
        pronunciation
            .candidates
            .first()
            .cloned()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(phoneme_index, planned)| {
                let mut features = planned.features.clone();
                if let Some(letter_index) = pronunciation.letter_indices.get(phoneme_index).copied()
                    && letter_index != NO_LETTER_INDEX
                {
                    add_letter_index_feature(&mut features, letter_index);
                    add_letter_name_feature(&mut features);
                }
                add_word_index_feature(&mut features, word_index);
                if let Some(part_of_speech) = pronunciation.part_of_speech {
                    add_part_of_speech_feature(&mut features, part_of_speech);
                }
                PhonemeToken {
                    phoneme: Spec::Known(planned.phoneme.clone()),
                    span: None,
                    features,
                    realized_as: Vec::new(),
                    confidence: confidence_for_status(pronunciation.status),
                    provenance: pronunciation.provenance.clone(),
                }
            })
            .collect()
    }

    fn phone_realizer(
        &self,
        variety: &LinguisticVariety,
        phonemes: &[PhonemeToken],
        careful_style: bool,
        syntax: &SentenceSyntaxAnalysis,
    ) -> Vec<PhoneToken> {
        realize_phonemes(
            variety,
            phonemes,
            &RealizationOptions {
                careful_style,
                phone_decomposition: PhoneDecompositionPolicy::KeepPhonemic,
                syntax: syntax.rule_context(),
            },
        )
    }

    fn output_provenance(&self, canonical_variety: &VarietyId) -> EvidenceProvenance {
        EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: format!(
                "{} variety data + staged pronunciation pipeline",
                canonical_variety.0
            ),
            version: Some("0.1".into()),
        }
    }

    fn run(&self, input: &PhonemicizeRequest) -> Result<PhonemicizeOutput, PhonemicizeError> {
        if input.text.trim().is_empty() {
            return Err(PhonemicizeError::EmptyInput);
        }

        let canonical_variety = self.canonical_variety_id(&input.variety)?;
        let variety = self.variety(&canonical_variety)?;
        let normalized_text = self.text_normalizer(&input.text, &canonical_variety);
        let mut words = self.orthographic_tokenizer(&normalized_text);
        mark_conjoined_letter_name_runs_for_variety(&mut words, &variety);
        let mut boundaries = self.boundary_extractor(&normalized_text, &words, &variety);
        let normalized_words = words
            .iter()
            .map(|word| word.normalized.clone())
            .collect::<Vec<_>>();
        let terminal = final_terminal(&boundaries);
        let syntax = self
            .syntax_analysis(&normalized_words, terminal, &variety)
            .unwrap_or_else(|| SentenceSyntaxAnalysis {
                tokens: normalized_words
                    .iter()
                    .enumerate()
                    .map(|(word_index, word)| crate::syntax::SyntaxToken {
                        word_index,
                        text: word.clone(),
                        pos: PartOfSpeech::Unknown,
                        prosodic_role: crate::syntax::ProsodicRole::Content,
                        syntactic_links: Vec::new(),
                    })
                    .collect(),
                link_parses: Vec::new(),
                raw_link_grammar_parses: Vec::new(),
                terminal,
            });
        self.annotate_boundaries(&mut boundaries, &words, &syntax, &variety);
        let prosody = prosody_from_boundaries(self, &boundaries, &words, &variety);
        let mut graphemes = Vec::with_capacity(words.len());
        let mut phonemes = Vec::new();
        let mut phones = Vec::new();
        let mut warnings = Vec::new();
        let style = input.style.clone().unwrap_or_default();
        let careful_style = style.careful_style;

        for (word_index, word) in words.iter().enumerate() {
            graphemes.push(GraphemeToken {
                grapheme: Spec::Known(GraphemeId(format!(
                    "{}.word.{}",
                    canonical_variety.0, word.normalized
                ))),
                text: word.text.clone(),
                span: Some(word.span),
                confidence: 1.0,
            });

            let context = TokenPronunciationContext {
                next_starts_with_vowelish: words
                    .get(word_index + 1)
                    .is_some_and(|next| self.next_word_starts_with_vowelish(next, &variety)),
                careful_style,
                part_of_speech: syntax.tokens.get(word_index).map(|token| token.pos),
                next_part_of_speech: syntax.tokens.get(word_index + 1).map(|token| token.pos),
            };
            let pronunciation = self.token_classifier(word, &variety, context);
            warnings.extend(pronunciation.warnings.clone());
            let mut word_phonemes =
                self.phoneme_planner(&canonical_variety, word_index, &pronunciation);
            let mut word_phones =
                self.phone_realizer(&variety, &word_phonemes, careful_style, &syntax);

            assign_realized_phones(&mut word_phonemes, &word_phones);
            if word_index > 0 {
                let has_pause_boundary = has_pause_boundary_after_word(&boundaries, word_index - 1);
                if !has_pause_boundary {
                    realize_connected_allophone_before_word(
                        &variety,
                        words
                            .get(word_index - 1)
                            .map(|word| word.normalized.as_str()),
                        &mut phonemes,
                        &mut phones,
                        word_phonemes.first(),
                        careful_style,
                    );
                }
                phones.push(boundary_phone_token());
                if !has_pause_boundary {
                    phones.extend(epenthetic_phones_between_words(
                        &variety,
                        phonemes.last(),
                        word_phonemes.first(),
                    ));
                }
            }
            phonemes.extend(word_phonemes);

            insert_letter_boundaries(&mut word_phones, &pronunciation.letter_break_offsets);
            phones.append(&mut word_phones);
        }
        let syllables = syllabify_phones(&phones, &variety);

        Ok(PhonemicizeOutput {
            text: input.text.clone(),
            variety: input.variety.clone(),
            graphemes,
            phonemes,
            phones,
            syllables,
            boundaries,
            prosody,
            syntax,
            warnings,
            provenance: self.output_provenance(&canonical_variety),
        })
    }

    fn next_word_starts_with_vowelish(
        &self,
        word: &WordToken,
        variety: &LinguisticVariety,
    ) -> bool {
        let candidate = self
            .token_classifier(
                word,
                variety,
                TokenPronunciationContext {
                    next_starts_with_vowelish: false,
                    careful_style: true,
                    part_of_speech: None,
                    next_part_of_speech: None,
                },
            )
            .candidates
            .first()
            .cloned()
            .unwrap_or_default();
        candidate
            .first()
            .is_some_and(|phoneme| planned_phoneme_is_vowel(variety, phoneme))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhonemicizeRequest {
    pub text: String,
    pub variety: VarietyId,
    pub style: Option<PhonemicizeStyle>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhonemicizeStyle {
    #[serde(default)]
    pub careful_style: bool,
}

impl Default for PhonemicizeStyle {
    fn default() -> Self {
        Self {
            careful_style: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemicizeOutput {
    pub text: String,
    pub variety: VarietyId,
    pub graphemes: Vec<GraphemeToken>,
    pub phonemes: Vec<PhonemeToken>,
    pub phones: Vec<PhoneToken>,
    pub syllables: Vec<Syllable>,
    #[serde(default)]
    pub boundaries: Vec<SpeechBoundaryToken>,
    #[serde(default)]
    pub prosody: ProsodyTrack,
    #[serde(default)]
    pub syntax: SentenceSyntaxAnalysis,
    #[serde(default)]
    pub warnings: Vec<PronunciationWarning>,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationWarning {
    pub token: String,
    pub kind: PronunciationWarningKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PronunciationWarningKind {
    GuessedWord,
    MixedAlphaNumeric,
    AcronymExpanded,
    WeakFormApplied,
    UnknownPronunciation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhonemicizeError {
    UnsupportedVariety { variety: VarietyId },
    EmptyInput,
}

impl fmt::Display for PhonemicizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVariety { variety } => {
                write!(
                    formatter,
                    "unsupported phonemicization variety `{}`",
                    variety.0
                )
            }
            Self::EmptyInput => formatter.write_str("cannot phonemicize empty input"),
        }
    }
}

impl std::error::Error for PhonemicizeError {}

#[derive(Debug, Clone, Default)]
pub struct VarietyDataPhonemicizer;

impl Phonemicizer for VarietyDataPhonemicizer {
    fn phonemicize(
        &self,
        input: &PhonemicizeRequest,
    ) -> Result<PhonemicizeOutput, PhonemicizeError> {
        self.run(input)
    }
}

impl PronunciationPipeline for VarietyDataPhonemicizer {
    fn canonical_variety_id(
        &self,
        requested_variety: &VarietyId,
    ) -> Result<VarietyId, PhonemicizeError> {
        canonical_variety_id(&requested_variety.0).ok_or_else(|| {
            PhonemicizeError::UnsupportedVariety {
                variety: requested_variety.clone(),
            }
        })
    }

    fn variety(
        &self,
        canonical_variety: &VarietyId,
    ) -> Result<LinguisticVariety, PhonemicizeError> {
        variety_by_code(&canonical_variety.0).ok_or_else(|| PhonemicizeError::UnsupportedVariety {
            variety: canonical_variety.clone(),
        })
    }

    fn orthographic_tokenizer(&self, text: &str) -> Vec<WordToken> {
        tokenize_words(text)
    }

    fn boundary_extractor(
        &self,
        text: &str,
        words: &[WordToken],
        variety: &LinguisticVariety,
    ) -> Vec<SpeechBoundaryToken> {
        boundary_tokens(text, words, variety)
    }

    fn weak_form_resolver(
        &self,
        word: &WordToken,
        variety: &LinguisticVariety,
        context: TokenPronunciationContext,
    ) -> Option<WordPronunciation> {
        data_driven_weak_form_pronunciation(word, variety, context)
    }

    fn token_classifier(
        &self,
        word: &WordToken,
        variety: &LinguisticVariety,
        context: TokenPronunciationContext,
    ) -> WordPronunciation {
        pronunciation_for_word(self, word, variety, context)
    }
}

#[derive(Debug, Clone)]
pub struct WordToken {
    pub text: String,
    pub normalized: String,
    pub kind: OrthographicTokenKind,
    pub span: TextSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrthographicTokenKind {
    Word,
    Acronym,
    MixedAlphaNumeric,
    LetterName,
    DigitName,
    Hyphenated(Vec<OrthographicToken>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrthographicToken {
    pub text: String,
    pub kind: Box<OrthographicTokenKind>,
}

fn tokenize_words(text: &str) -> Vec<WordToken> {
    let mut words = Vec::new();
    let mut start = None;
    for (byte_index, character) in text.char_indices() {
        if is_word_chunk_character(character) {
            start.get_or_insert(byte_index);
            continue;
        }

        if let Some(start_byte) = start.take() {
            push_word_chunk(text, start_byte, byte_index, &mut words);
        }
    }

    if let Some(start_byte) = start {
        push_word_chunk(text, start_byte, text.len(), &mut words);
    }

    mark_spaced_letter_name_runs(&mut words);
    mark_contextual_initialisms(text, &mut words);
    words
}

fn is_word_chunk_character(character: char) -> bool {
    character.is_alphanumeric()
        || is_combining_word_mark(character)
        || is_apostrophe(character)
        || character == '-'
}

fn is_combining_word_mark(character: char) -> bool {
    matches!(
        character,
        '\u{0300}'..='\u{036F}' | '\u{0900}'..='\u{094D}' | '\u{0951}'..='\u{0957}'
    )
}

fn is_apostrophe(character: char) -> bool {
    matches!(character, '\'' | '’' | '‘' | 'ʼ')
}

fn push_word_chunk(text: &str, start_byte: usize, end_byte: usize, words: &mut Vec<WordToken>) {
    let surface = &text[start_byte..end_byte];
    if should_preserve_hyphenated_word_chunk(surface) {
        push_word(text, start_byte, end_byte, words);
        return;
    }

    let mut part_start = None;
    for (offset, character) in text[start_byte..end_byte].char_indices() {
        let byte_index = start_byte + offset;
        if character == '-' {
            if let Some(part_start_byte) = part_start.take() {
                push_camelcase_word_parts(text, part_start_byte, byte_index, words);
            }
            continue;
        }

        part_start.get_or_insert(byte_index);
    }

    if let Some(part_start_byte) = part_start {
        push_camelcase_word_parts(text, part_start_byte, end_byte, words);
    }
}

fn push_camelcase_word_parts(
    text: &str,
    start_byte: usize,
    end_byte: usize,
    words: &mut Vec<WordToken>,
) {
    let mut part_start = start_byte;
    let mut previous = None;
    let mut iterator = text[start_byte..end_byte].char_indices().peekable();
    while let Some((offset, character)) = iterator.next() {
        let byte_index = start_byte + offset;
        if let Some(previous_character) = previous
            && should_split_camelcase_part(previous_character, character, iterator.peek())
        {
            push_word(text, part_start, byte_index, words);
            part_start = byte_index;
        }
        previous = Some(character);
    }

    push_word(text, part_start, end_byte, words);
}

fn should_split_camelcase_part(
    previous: char,
    current: char,
    next: Option<&(usize, char)>,
) -> bool {
    previous.is_lowercase()
        && current.is_uppercase()
        && next.is_some_and(|(_, next)| next.is_uppercase())
}

fn push_word(text: &str, start_byte: usize, end_byte: usize, words: &mut Vec<WordToken>) {
    let surface = &text[start_byte..end_byte];
    if should_split_mixed_surface_into_units(surface) {
        push_orthographic_unit_words(text, start_byte, end_byte, words);
        return;
    }

    let start_char = text[..start_byte].chars().count();
    let end_char = start_char + surface.chars().count();
    let normalized = normalize_surface_word(surface);
    if normalized.is_empty() {
        return;
    }

    words.push(WordToken {
        text: surface.to_string(),
        normalized,
        kind: classify_surface_word(surface),
        span: TextSpan {
            start_char,
            end_char,
        },
    });
}

fn should_split_mixed_surface_into_units(surface: &str) -> bool {
    let has_alpha = surface.chars().any(char::is_alphabetic);
    let has_digit = surface.chars().any(|character| character.is_ascii_digit());
    has_alpha
        && has_digit
        && surface
            .chars()
            .filter(|character| character.is_alphabetic())
            .all(|character| character.is_uppercase())
}

fn should_preserve_hyphenated_word_chunk(surface: &str) -> bool {
    let parts = split_hyphenated_surface(surface);
    parts.len() > 1
        && parts.iter().all(|part| {
            part.chars().any(char::is_alphabetic)
                && part
                    .chars()
                    .all(|character| character.is_alphabetic() || is_apostrophe(character))
        })
}

fn push_orthographic_unit_words(
    text: &str,
    start_byte: usize,
    end_byte: usize,
    words: &mut Vec<WordToken>,
) {
    for (offset, character) in text[start_byte..end_byte].char_indices() {
        if !character.is_alphanumeric() {
            continue;
        }
        let byte_index = start_byte + offset;
        let start_char = text[..byte_index].chars().count();
        let kind = if character.is_ascii_digit() {
            OrthographicTokenKind::DigitName
        } else {
            OrthographicTokenKind::LetterName
        };
        words.push(WordToken {
            text: character.to_string(),
            normalized: character.to_lowercase().collect(),
            kind,
            span: TextSpan {
                start_char,
                end_char: start_char + 1,
            },
        });
    }
}

fn normalize_surface_word(surface: &str) -> String {
    surface
        .trim_matches(|character: char| !character.is_alphabetic())
        .chars()
        .flat_map(|character| {
            if is_apostrophe(character) {
                "'".chars().collect::<Vec<_>>()
            } else {
                character.to_lowercase().collect()
            }
        })
        .collect()
}

fn boundary_tokens(
    text: &str,
    words: &[WordToken],
    variety: &LinguisticVariety,
) -> Vec<SpeechBoundaryToken> {
    if words.is_empty() {
        return Vec::new();
    }

    let text_len_chars = text.chars().count();
    let mut boundaries = Vec::new();
    for (index, word) in words.iter().enumerate() {
        let next_start = words
            .get(index + 1)
            .map(|next| next.span.start_char)
            .unwrap_or(text_len_chars);
        let next_word = words.get(index + 1);
        if let Some(boundary) =
            punctuation_boundary_after_word(text, word, index, next_start, next_word, variety)
        {
            boundaries.push(boundary);
        } else if index + 1 < words.len() {
            boundaries.push(SpeechBoundaryToken {
                kind: BoundaryKind::Word,
                after_grapheme_index: index,
                span: None,
                terminal: None,
                pause: None,
            });
        }
    }

    if !boundaries
        .iter()
        .any(|boundary| boundary.terminal.is_some())
    {
        boundaries.push(SpeechBoundaryToken {
            kind: BoundaryKind::Phrase,
            after_grapheme_index: words.len() - 1,
            span: None,
            terminal: Some(TerminalPunctuation::Period),
            pause: None,
        });
    }

    boundaries
}

fn final_terminal(boundaries: &[SpeechBoundaryToken]) -> Option<TerminalPunctuation> {
    boundaries
        .iter()
        .rev()
        .find_map(|boundary| boundary.terminal)
}

fn prosody_from_boundaries(
    pipeline: &(impl PronunciationPipeline + ?Sized),
    boundaries: &[SpeechBoundaryToken],
    words: &[WordToken],
    variety: &LinguisticVariety,
) -> ProsodyTrack {
    let mut prosody = ProsodyTrack::default();
    let mut sentence_start_word_index = 0;
    for boundary in boundaries {
        let Some(kind) = pipeline.prosodic_label_for_boundary(
            boundary,
            words,
            sentence_start_word_index,
            variety,
        ) else {
            if boundary.terminal.is_some() {
                sentence_start_word_index = boundary.after_grapheme_index.saturating_add(1);
            }
            continue;
        };
        prosody.labels.push(ProsodicLabel {
            span: TimeSpan {
                start_s: 0.0,
                end_s: 0.0,
            },
            kind,
            confidence: if boundary.span.is_some() { 0.9 } else { 0.55 },
        });
        if boundary.terminal.is_some() {
            sentence_start_word_index = boundary.after_grapheme_index.saturating_add(1);
        }
    }
    prosody
}

fn generic_prosodic_label_for_boundary(
    boundary: &SpeechBoundaryToken,
) -> Option<ProsodicLabelKind> {
    match (boundary.terminal, boundary.pause) {
        (Some(TerminalPunctuation::Question), _) => Some(ProsodicLabelKind::QuestionRise),
        (Some(TerminalPunctuation::Period | TerminalPunctuation::Exclamation), _) => {
            Some(ProsodicLabelKind::FinalFall)
        }
        (None, Some(PauseKind::AlternativeQuestionRise)) => {
            Some(ProsodicLabelKind::AlternativeQuestionRise)
        }
        (None, Some(PauseKind::Comma)) => Some(ProsodicLabelKind::ContinuationRise),
        _ => None,
    }
}

fn annotate_alternative_question_boundaries(
    boundaries: &mut Vec<SpeechBoundaryToken>,
    words: &[WordToken],
    syntax: &SentenceSyntaxAnalysis,
    variety: &LinguisticVariety,
) {
    if final_terminal(boundaries) != Some(TerminalPunctuation::Question) {
        return;
    }
    let Some(profile) = variety.question_contours.as_ref() else {
        return;
    };
    if !words
        .first()
        .is_some_and(|word| profile.yes_no_openers.contains(&word.normalized))
    {
        return;
    }
    let normalized_words = words
        .iter()
        .map(|word| word.normalized.as_str())
        .collect::<Vec<_>>();
    let Some(first_option_index) =
        alternative_question_first_option_index(&normalized_words, syntax, profile)
    else {
        return;
    };

    if let Some(boundary) = boundaries
        .iter_mut()
        .find(|boundary| boundary.after_grapheme_index == first_option_index)
    {
        if boundary.terminal.is_none() {
            boundary.kind = BoundaryKind::Phrase;
            boundary.pause = Some(PauseKind::AlternativeQuestionRise);
        }
    } else {
        boundaries.push(SpeechBoundaryToken {
            kind: BoundaryKind::Phrase,
            after_grapheme_index: first_option_index,
            span: None,
            terminal: None,
            pause: Some(PauseKind::AlternativeQuestionRise),
        });
        boundaries.sort_by_key(|boundary| boundary.after_grapheme_index);
    }
}

fn alternative_question_first_option_index(
    words: &[&str],
    syntax: &SentenceSyntaxAnalysis,
    profile: &crate::variety::QuestionContourProfile,
) -> Option<usize> {
    let parse = syntax.primary_parse()?;
    words
        .iter()
        .enumerate()
        .filter(|(index, word)| {
            profile
                .alternative_coordinators
                .iter()
                .any(|coordinator| coordinator == *word)
                && *index > 0
                && index + 1 < words.len()
        })
        .find_map(|(or_index, _)| {
            let has_linked_options = parse.links.iter().any(|link| {
                link.kind == crate::syntax::SyntacticLinkKind::Coordination
                    && link.left + 2 == link.right
                    && link.left + 1 == or_index
            });
            has_linked_options.then_some(or_index - 1)
        })
}

fn words_in_sentence<'a>(
    words: &'a [WordToken],
    sentence_start_word_index: usize,
    boundary: &SpeechBoundaryToken,
) -> &'a [WordToken] {
    let start = sentence_start_word_index.min(words.len());
    let end = boundary
        .after_grapheme_index
        .saturating_add(1)
        .min(words.len());
    if start >= end {
        &[]
    } else {
        &words[start..end]
    }
}

fn prosodic_label_for_question(
    words: &[WordToken],
    variety: &LinguisticVariety,
) -> ProsodicLabelKind {
    let Some(profile) = variety.question_contours.as_ref() else {
        return ProsodicLabelKind::QuestionRise;
    };
    match question_contour(words, profile) {
        QuestionContour::Rising => ProsodicLabelKind::QuestionRise,
        QuestionContour::AlternativeFall => ProsodicLabelKind::AlternativeQuestionFall,
        QuestionContour::FinalFall => ProsodicLabelKind::FinalFall,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuestionContour {
    Rising,
    AlternativeFall,
    FinalFall,
}

fn question_contour(
    words: &[WordToken],
    profile: &crate::variety::QuestionContourProfile,
) -> QuestionContour {
    let Some(first) = words.first().map(|word| word.normalized.as_str()) else {
        return QuestionContour::Rising;
    };

    if has_alternative_question_coordination(words, profile) {
        return QuestionContour::AlternativeFall;
    }
    if profile.wh_openers.iter().any(|opener| opener == first) {
        return QuestionContour::FinalFall;
    }

    QuestionContour::Rising
}

fn has_alternative_question_coordination(
    words: &[WordToken],
    profile: &crate::variety::QuestionContourProfile,
) -> bool {
    if has_paired_alternative_coordination(words, profile) {
        return true;
    }
    if !words
        .first()
        .is_some_and(|word| profile.yes_no_openers.contains(&word.normalized))
    {
        return false;
    }

    words.iter().enumerate().skip(1).any(|(index, word)| {
        profile.alternative_coordinators.contains(&word.normalized) && index + 1 < words.len()
    })
}

fn has_paired_alternative_coordination(
    words: &[WordToken],
    profile: &crate::variety::QuestionContourProfile,
) -> bool {
    let Some(either_index) = words.iter().position(|word| {
        profile
            .paired_alternative_openers
            .contains(&word.normalized)
    }) else {
        return false;
    };
    words
        .iter()
        .skip(either_index + 1)
        .any(|word| profile.alternative_coordinators.contains(&word.normalized))
}

fn punctuation_boundary_after_word(
    text: &str,
    word: &WordToken,
    word_index: usize,
    next_start_char: usize,
    next_word: Option<&WordToken>,
    variety: &LinguisticVariety,
) -> Option<SpeechBoundaryToken> {
    let mut found = None;
    for (char_index, character) in text.chars().enumerate() {
        if char_index < word.span.end_char || char_index >= next_start_char {
            continue;
        }

        let mut terminal = match character {
            '.' | '…' => Some(TerminalPunctuation::Period),
            '?' => Some(TerminalPunctuation::Question),
            '!' => Some(TerminalPunctuation::Exclamation),
            _ => None,
        };

        if terminal == Some(TerminalPunctuation::Period) {
            if let Some(profile) = variety.punctuation.as_ref()
                && profile
                    .period_abbreviations
                    .iter()
                    .any(|abbr| abbr == &word.normalized)
            {
                if profile
                    .title_abbreviations
                    .iter()
                    .any(|abbr| abbr == &word.normalized)
                {
                    if next_word.is_some() {
                        terminal = None;
                    }
                } else {
                    if let Some(next) = next_word {
                        let next_first_char = next.text.chars().next();
                        let next_is_uppercase = next_first_char.map_or(false, |c| c.is_uppercase());
                        if !next_is_uppercase {
                            terminal = None;
                        } else if profile
                            .ambiguous_period_abbreviations
                            .iter()
                            .any(|abbr| abbr == &word.normalized)
                        {
                            let next_word_lower = next.normalized.as_str();
                            let is_sentence_starter = profile
                                .sentence_starter_words_after_ambiguous_abbreviation
                                .iter()
                                .any(|candidate| candidate == next_word_lower);
                            if !is_sentence_starter {
                                terminal = None;
                            }
                        }
                    }
                }
            }
        }

        let pause = match character {
            ',' | ';' | ':' => Some(PauseKind::Comma),
            _ => None,
        };
        if terminal.is_some() || pause.is_some() {
            found = Some(SpeechBoundaryToken {
                kind: BoundaryKind::Phrase,
                after_grapheme_index: word_index,
                span: Some(TextSpan {
                    start_char: char_index,
                    end_char: char_index + 1,
                }),
                terminal,
                pause,
            });
        }
    }
    found
}

fn classify_surface_word(surface: &str) -> OrthographicTokenKind {
    let has_alpha = surface.chars().any(char::is_alphabetic);
    let has_digit = surface.chars().any(|character| character.is_ascii_digit());
    if surface.contains('-') {
        return OrthographicTokenKind::Hyphenated(
            split_hyphenated_surface(surface)
                .into_iter()
                .filter_map(|part| {
                    let normalized = normalize_surface_word(part);
                    (!normalized.is_empty()).then(|| OrthographicToken {
                        text: part.to_string(),
                        kind: Box::new(classify_non_hyphenated_surface_word(part)),
                    })
                })
                .collect(),
        );
    }
    classify_non_hyphenated_surface_word(surface)
}

fn classify_non_hyphenated_surface_word(surface: &str) -> OrthographicTokenKind {
    let has_alpha = surface.chars().any(char::is_alphabetic);
    let has_digit = surface.chars().any(|character| character.is_ascii_digit());
    if has_alpha && has_digit {
        OrthographicTokenKind::MixedAlphaNumeric
    } else {
        OrthographicTokenKind::Word
    }
}

fn split_hyphenated_surface(surface: &str) -> Vec<&str> {
    surface.split('-').filter(|part| !part.is_empty()).collect()
}

fn mark_spaced_letter_name_runs(words: &mut [WordToken]) {
    let mut index = 0;
    while index < words.len() {
        if !is_letter_name_candidate_word(&words[index]) {
            index += 1;
            continue;
        }

        let run_start = index;
        while index < words.len() && is_letter_name_candidate_word(&words[index]) {
            index += 1;
        }

        if index - run_start < 2 {
            continue;
        }

        for word in &mut words[run_start..index] {
            word.kind = OrthographicTokenKind::LetterName;
        }
    }
}

fn mark_conjoined_letter_name_runs_for_variety(
    words: &mut [WordToken],
    variety: &LinguisticVariety,
) {
    if words.len() < 3 {
        return;
    }
    let Some(orthography) = variety.orthography.as_ref() else {
        return;
    };
    for index in 1..words.len() - 1 {
        if !orthography
            .initialism_joiners
            .iter()
            .any(|joiner| joiner == &words[index].normalized)
        {
            continue;
        }
        if is_letter_name_candidate_word(&words[index - 1])
            && is_letter_name_candidate_word(&words[index + 1])
        {
            words[index - 1].kind = OrthographicTokenKind::LetterName;
            words[index + 1].kind = OrthographicTokenKind::LetterName;
        }
    }
}

fn is_letter_name_candidate_word(word: &WordToken) -> bool {
    if !matches!(
        word.kind,
        OrthographicTokenKind::Word | OrthographicTokenKind::LetterName
    ) {
        return false;
    }
    let mut characters = word.text.chars();
    let Some(character) = characters.next() else {
        return false;
    };
    characters.next().is_none()
        && character.is_alphabetic()
        && (character.is_uppercase() || !character.is_lowercase())
}

fn mark_contextual_initialisms(text: &str, words: &mut [WordToken]) {
    if !text.chars().any(char::is_lowercase) {
        return;
    }
    for word in words {
        if matches!(word.kind, OrthographicTokenKind::Word)
            && is_short_uppercase_initialism_surface(&word.text)
        {
            word.kind = OrthographicTokenKind::Acronym;
        }
    }
}

#[derive(Debug, Clone)]
pub struct WordPronunciation {
    pub candidates: Vec<Vec<PlannedPhoneme>>,
    pub status: PronunciationStatus,
    pub provenance: EvidenceProvenance,
    pub warnings: Vec<PronunciationWarning>,
    pub letter_break_offsets: Vec<usize>,
    pub letter_indices: Vec<usize>,
    pub part_of_speech: Option<PartOfSpeech>,
}

#[derive(Debug, Clone)]
pub struct PlannedPhoneme {
    pub phoneme: PhonemeId,
    pub features: FeatureBundle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenPronunciationContext {
    pub next_starts_with_vowelish: bool,
    pub careful_style: bool,
    pub part_of_speech: Option<PartOfSpeech>,
    pub next_part_of_speech: Option<PartOfSpeech>,
}

fn planned_candidates_from_notation(
    variety: &LinguisticVariety,
    candidates: Vec<Vec<String>>,
    notation: PronunciationNotation,
) -> Vec<Vec<PlannedPhoneme>> {
    candidates
        .into_iter()
        .map(|candidate| planned_candidate_from_notation(variety, candidate, notation))
        .filter(|candidate| !candidate.is_empty())
        .collect()
}

fn planned_candidate_from_notation(
    variety: &LinguisticVariety,
    candidate: Vec<String>,
    notation: PronunciationNotation,
) -> Vec<PlannedPhoneme> {
    notation::parse_pronunciation_candidate(variety, &candidate, notation)
        .into_iter()
        .map(|token| PlannedPhoneme {
            phoneme: token.phoneme,
            features: token.features,
        })
        .collect()
}

fn source_pronunciation_notation(name: Option<&str>) -> Option<PronunciationNotation> {
    notation::pronunciation_notation(name)
}

fn planned_candidate_from_source_pronunciation(
    variety: &LinguisticVariety,
    symbols: &[String],
    notation: Option<&str>,
) -> Vec<PlannedPhoneme> {
    let Some(notation) = source_pronunciation_notation(notation) else {
        return Vec::new();
    };
    planned_candidate_from_notation(variety, symbols.to_vec(), notation)
}

fn fallback_pronunciation(
    word: &WordToken,
    variety: &LinguisticVariety,
    context: TokenPronunciationContext,
) -> Option<WordPronunciation> {
    let fallback = unknown_pronunciation_fallback()?;
    let prediction = fallback(&word.normalized, &variety.id.0)?;
    let candidate =
        planned_candidate_from_notation(variety, vec![prediction], PronunciationNotation::Ipa);
    if candidate.is_empty() {
        return None;
    }

    Some(WordPronunciation {
        candidates: vec![candidate],
        status: PronunciationStatus::Guessed,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Inference,
            method: "unknown-word pronunciation fallback".into(),
            version: Some("0.1".into()),
        },
        warnings: vec![PronunciationWarning {
            token: word.text.clone(),
            kind: PronunciationWarningKind::GuessedWord,
            message: format!("guessed pronunciation: {}", word.text),
        }],
        letter_break_offsets: Vec::new(),
        letter_indices: Vec::new(),
        part_of_speech: context.part_of_speech,
    })
}

fn planned_phoneme_is_vowel(variety: &LinguisticVariety, planned: &PlannedPhoneme) -> bool {
    variety
        .phonemes
        .phonemes
        .get(&planned.phoneme)
        .is_some_and(|phoneme| {
            phoneme
                .features
                .values
                .get(&FeatureId("phonology.major".into()))
                == Some(&Spec::Known(FeatureValue::Category("vowel".into())))
        })
}

fn pronunciation_from_lexicon_id(
    lexicon_id: &str,
    word: &WordToken,
    variety: &LinguisticVariety,
    context: TokenPronunciationContext,
) -> Option<WordPronunciation> {
    let adapter = lexicons::adapter_for_id(lexicon_id)?;
    pronunciation_from_lexicon_adapter(adapter, word, variety, context)
}

fn pronunciation_from_declared_lexicons(
    word: &WordToken,
    variety: &LinguisticVariety,
    context: TokenPronunciationContext,
) -> Option<WordPronunciation> {
    variety
        .pronunciation_lexicons
        .iter()
        .find_map(|lexicon| pronunciation_from_lexicon_id(lexicon, word, variety, context))
}

fn planned_candidate_from_orthography_profile(
    normalized: &str,
    variety: &LinguisticVariety,
    context: TokenPronunciationContext,
) -> Vec<PlannedPhoneme> {
    if let Some(ipa) = variety
        .orthography_pronunciation
        .and_then(|rules| rules.synthesize_ipa)
        .and_then(|synthesize| synthesize(normalized, variety, context.part_of_speech))
    {
        return planned_candidate_from_variety_ipa(&ipa, variety);
    }

    planned_candidate_from_variety_aliases(normalized, variety)
}

fn pronunciation_from_lexicon_adapter(
    adapter: lexicons::LexiconAdapter,
    word: &WordToken,
    variety: &LinguisticVariety,
    context: TokenPronunciationContext,
) -> Option<WordPronunciation> {
    let entry = (adapter.lookup)(&word.normalized);
    if entry.candidates.is_empty() {
        return None;
    }
    let selection =
        choose_context_sensitive_candidates(variety, &entry.lookup, entry.candidates, context);
    let candidates =
        planned_candidates_from_notation(variety, selection.candidates, adapter.notation);
    if candidates.is_empty() {
        return None;
    }
    Some(WordPronunciation {
        candidates,
        status: entry.status,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Lexicon,
            method: pronunciation_lookup_method(
                adapter.id,
                entry.status,
                entry.source,
                context.part_of_speech,
                selection.applied_pos,
            ),
            version: Some(entry.source.into()),
        },
        warnings: Vec::new(),
        letter_break_offsets: Vec::new(),
        letter_indices: Vec::new(),
        part_of_speech: context.part_of_speech,
    })
}

fn planned_candidate_from_variety_ipa(
    ipa: &str,
    variety: &LinguisticVariety,
) -> Vec<PlannedPhoneme> {
    planned_candidate_from_notation(variety, vec![ipa.to_string()], PronunciationNotation::Ipa)
}

fn planned_candidate_from_variety_aliases(
    normalized: &str,
    variety: &LinguisticVariety,
) -> Vec<PlannedPhoneme> {
    let aliases = notation::phoneme_aliases_by_length(variety);
    let chars = normalized.chars().collect::<Vec<_>>();
    let mut index = 0usize;
    let mut candidate = Vec::new();
    while index < chars.len() {
        let rest = chars[index..].iter().collect::<String>();
        if let Some((phoneme, consumed)) = aliases.iter().find_map(|(alias, phoneme)| {
            rest.starts_with(alias)
                .then_some((phoneme, alias.chars().count()))
        }) {
            candidate.push(planned_phoneme_from_inventory(phoneme));
            index += consumed;
        } else {
            return Vec::new();
        }
    }
    candidate
}

fn planned_phoneme_from_inventory(phoneme: &crate::phonology::Phoneme) -> PlannedPhoneme {
    PlannedPhoneme {
        phoneme: phoneme.id.clone(),
        features: phoneme.features.clone(),
    }
}

fn pronunciation_for_word(
    pipeline: &(impl PronunciationPipeline + ?Sized),
    word: &WordToken,
    variety: &LinguisticVariety,
    context: TokenPronunciationContext,
) -> WordPronunciation {
    match &word.kind {
        OrthographicTokenKind::Acronym => {
            return acronym_pronunciation(word.text.as_str(), variety);
        }
        OrthographicTokenKind::MixedAlphaNumeric => {
            return mixed_alphanumeric_pronunciation(word, variety);
        }
        OrthographicTokenKind::LetterName => {
            return orthographic_unit_pronunciation(
                word,
                variety,
                OrthographicUnitKind::LetterName,
                Some(0),
            );
        }
        OrthographicTokenKind::DigitName => {
            return orthographic_unit_pronunciation(
                word,
                variety,
                OrthographicUnitKind::DigitName,
                None,
            );
        }
        OrthographicTokenKind::Word | OrthographicTokenKind::Hyphenated(_) => {}
    }

    if let Some(pronunciation) = pipeline.weak_form_resolver(word, variety, context) {
        return pronunciation;
    }

    if let Some(pronunciation) = pronunciation_from_declared_lexicons(word, variety, context) {
        return pronunciation;
    }

    if let OrthographicTokenKind::Hyphenated(parts) = &word.kind
        && let Some(pronunciation) =
            hyphenated_pronunciation_from_parts(pipeline, word, parts, variety, context)
    {
        return pronunciation;
    }

    if pipeline.uses_generic_orthography_before_unknown() {
        let candidate =
            planned_candidate_from_orthography_profile(&word.normalized, variety, context);
        if !candidate.is_empty() {
            return WordPronunciation {
                candidates: vec![candidate],
                status: PronunciationStatus::Exact,
                provenance: EvidenceProvenance {
                    source: EvidenceSource::Rule,
                    method: format!("{} orthography profile pronunciation", variety.id.0),
                    version: Some("0.1".into()),
                },
                warnings: Vec::new(),
                letter_break_offsets: Vec::new(),
                letter_indices: Vec::new(),
                part_of_speech: context.part_of_speech,
            };
        }
    }

    pipeline.unknown_word_pronunciation(word, variety, context)
}

fn hyphenated_pronunciation_from_parts(
    pipeline: &(impl PronunciationPipeline + ?Sized),
    word: &WordToken,
    parts: &[OrthographicToken],
    variety: &LinguisticVariety,
    context: TokenPronunciationContext,
) -> Option<WordPronunciation> {
    if parts.len() < 2 {
        return None;
    }

    let mut candidate = Vec::new();
    let mut break_offsets = Vec::new();
    let mut warnings = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        let normalized = normalize_surface_word(&part.text);
        if normalized.is_empty() {
            return None;
        }
        let part_word = WordToken {
            text: part.text.clone(),
            normalized,
            kind: (*part.kind).clone(),
            span: word.span,
        };
        let mut part_context = context;
        part_context.part_of_speech = None;
        part_context.next_part_of_speech = None;
        part_context.next_starts_with_vowelish = parts
            .get(index + 1)
            .is_some_and(|next| hyphenated_part_starts_with_vowelish(next, variety));
        let part_pronunciation =
            pronunciation_for_word(pipeline, &part_word, variety, part_context);
        let part_candidate = part_pronunciation.candidates.first()?;
        if part_candidate.is_empty() {
            return None;
        }
        candidate.extend(part_candidate.clone());
        warnings.extend(part_pronunciation.warnings);
        if index + 1 < parts.len() {
            break_offsets.push(candidate.len());
        }
    }

    Some(WordPronunciation {
        candidates: vec![candidate],
        status: PronunciationStatus::Guessed,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: "hyphenated compound pronunciation from segment pronunciations".into(),
            version: Some("0.1".into()),
        },
        warnings,
        letter_break_offsets: break_offsets,
        letter_indices: Vec::new(),
        part_of_speech: context.part_of_speech,
    })
}

fn hyphenated_part_starts_with_vowelish(
    part: &OrthographicToken,
    variety: &LinguisticVariety,
) -> bool {
    let normalized = normalize_surface_word(&part.text);
    if normalized.is_empty() {
        return false;
    }
    let word = WordToken {
        text: part.text.clone(),
        normalized,
        kind: (*part.kind).clone(),
        span: TextSpan {
            start_char: 0,
            end_char: part.text.chars().count(),
        },
    };
    let context = TokenPronunciationContext {
        next_starts_with_vowelish: false,
        careful_style: true,
        part_of_speech: None,
        next_part_of_speech: None,
    };
    pronunciation_from_declared_lexicons(&word, variety, context)
        .or_else(|| {
            let candidate =
                planned_candidate_from_orthography_profile(&word.normalized, variety, context);
            (!candidate.is_empty()).then(|| WordPronunciation {
                candidates: vec![candidate],
                status: PronunciationStatus::Guessed,
                provenance: EvidenceProvenance {
                    source: EvidenceSource::Rule,
                    method: "orthography profile pronunciation".into(),
                    version: Some("0.1".into()),
                },
                warnings: Vec::new(),
                letter_break_offsets: Vec::new(),
                letter_indices: Vec::new(),
                part_of_speech: None,
            })
        })
        .and_then(|pronunciation| pronunciation.candidates.first().cloned())
        .and_then(|candidate| candidate.first().cloned())
        .is_some_and(|phoneme| planned_phoneme_is_vowel(variety, &phoneme))
}

fn missing_pronunciation(
    word: &WordToken,
    context: TokenPronunciationContext,
    method: &str,
) -> WordPronunciation {
    WordPronunciation {
        candidates: Vec::new(),
        status: PronunciationStatus::Missing,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Unknown,
            method: method.into(),
            version: Some("0.1".into()),
        },
        warnings: vec![PronunciationWarning {
            token: word.text.clone(),
            kind: PronunciationWarningKind::UnknownPronunciation,
            message: format!("unknown pronunciation: {}", word.text),
        }],
        letter_break_offsets: Vec::new(),
        letter_indices: Vec::new(),
        part_of_speech: context.part_of_speech,
    }
}

fn is_short_uppercase_initialism_surface(surface: &str) -> bool {
    let mut alpha_count = 0usize;
    for character in surface
        .chars()
        .filter(|character| character.is_alphabetic())
    {
        if !character.is_uppercase() {
            return false;
        }
        alpha_count += 1;
    }
    (2..=3).contains(&alpha_count)
}

#[derive(Debug)]
struct CandidateSelection {
    candidates: Vec<Vec<String>>,
    applied_pos: bool,
}

fn choose_context_sensitive_candidates(
    variety: &LinguisticVariety,
    lookup: &str,
    candidates: Vec<Vec<String>>,
    context: TokenPronunciationContext,
) -> CandidateSelection {
    let Some(rule) = pronunciation_selection_rule(variety, lookup, context) else {
        return CandidateSelection {
            candidates,
            applied_pos: false,
        };
    };
    choose_matching_candidate(&candidates, &rule.source_pronunciation, true).unwrap_or(
        CandidateSelection {
            candidates,
            applied_pos: false,
        },
    )
}

fn choose_matching_candidate(
    candidates: &[Vec<String>],
    symbols: &[String],
    applied_pos: bool,
) -> Option<CandidateSelection> {
    let Some(position) = candidates
        .iter()
        .position(|candidate| candidate_matches_symbols(candidate, symbols))
    else {
        return None;
    };

    let mut selected = candidates.to_vec();
    if position > 0 {
        let preferred = selected.remove(position);
        selected.insert(0, preferred);
    }
    Some(CandidateSelection {
        candidates: selected,
        applied_pos,
    })
}

fn pronunciation_selection_rule<'a>(
    variety: &'a LinguisticVariety,
    lookup: &str,
    context: TokenPronunciationContext,
) -> Option<&'a crate::variety::PronunciationSelectionRule> {
    let part_of_speech = context.part_of_speech.map(canonical_pronunciation_pos);
    variety.pronunciation_selection_rules.iter().find(|rule| {
        rule.lexical_item == lookup
            && rule
                .part_of_speech
                .map(canonical_pronunciation_pos)
                .is_none_or(|expected| Some(expected) == part_of_speech)
            && rule
                .next_part_of_speech
                .is_none_or(|expected| Some(expected) == context.next_part_of_speech)
            && !rule.source_pronunciation.is_empty()
    })
}

fn canonical_pronunciation_pos(part_of_speech: PartOfSpeech) -> PartOfSpeech {
    match part_of_speech {
        PartOfSpeech::Auxiliary => PartOfSpeech::Verb,
        other => other,
    }
}

fn candidate_matches_symbols(candidate: &[String], symbols: &[String]) -> bool {
    candidate == symbols
        || (candidate.len() == symbols.len()
            && candidate.iter().zip(symbols).all(|(candidate, source)| {
                crate::data::lexicons::cmudict_rp::adapted_symbol_matches_source(candidate, source)
            }))
}

fn weak_form_rule_applies(
    rule: &WeakFormRule,
    normalized: &str,
    context: TokenPronunciationContext,
) -> bool {
    if rule.lexical_item != normalized {
        return false;
    }
    if rule.style == WeakFormStyleContext::CasualOnly && context.careful_style {
        return false;
    }
    if !weak_form_pos_allows(context.part_of_speech) {
        return false;
    }
    match rule.following {
        WeakFormFollowingContext::Any => true,
        WeakFormFollowingContext::BeforeVowelish => context.next_starts_with_vowelish,
        WeakFormFollowingContext::BeforeConsonantish => !context.next_starts_with_vowelish,
    }
}

fn weak_form_pos_allows(part_of_speech: Option<PartOfSpeech>) -> bool {
    part_of_speech.is_none_or(|part_of_speech| {
        matches!(
            part_of_speech,
            PartOfSpeech::Auxiliary
                | PartOfSpeech::Determiner
                | PartOfSpeech::Preposition
                | PartOfSpeech::Pronoun
                | PartOfSpeech::Conjunction
                | PartOfSpeech::Particle
                | PartOfSpeech::Unknown
        )
    })
}

fn data_driven_weak_form_pronunciation(
    word: &WordToken,
    variety: &LinguisticVariety,
    context: TokenPronunciationContext,
) -> Option<WordPronunciation> {
    variety
        .weak_forms
        .iter()
        .find(|rule| weak_form_rule_applies(rule, &word.normalized, context))
        .map(|rule| weak_form_pronunciation(rule, variety))
}

fn weak_form_pronunciation(rule: &WeakFormRule, variety: &LinguisticVariety) -> WordPronunciation {
    let candidate = if rule.source_pronunciation.is_empty() {
        rule.pronunciation
            .iter()
            .filter_map(|id| {
                variety
                    .phonemes
                    .phonemes
                    .get(id)
                    .map(planned_phoneme_from_inventory)
            })
            .collect()
    } else {
        planned_candidate_from_source_pronunciation(
            variety,
            &rule.source_pronunciation,
            rule.source_pronunciation_notation.as_deref(),
        )
    };
    let method = format!("variety weak form: {}", rule.id.replace('_', " "));
    WordPronunciation {
        candidates: vec![candidate],
        status: PronunciationStatus::Exact,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: method.clone(),
            version: Some("0.1".into()),
        },
        warnings: Vec::new(),
        letter_break_offsets: Vec::new(),
        letter_indices: Vec::new(),
        part_of_speech: None,
    }
}

fn acronym_pronunciation(surface: &str, variety: &LinguisticVariety) -> WordPronunciation {
    let (candidate, letter_break_offsets, letter_indices) =
        letter_name_sequence(surface.chars(), variety);
    let expanded = candidate
        .iter()
        .map(|phoneme| phoneme_display_symbol(&phoneme.phoneme))
        .collect::<Vec<_>>()
        .join(" ");
    WordPronunciation {
        candidates: vec![candidate],
        status: PronunciationStatus::Exact,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: "variety acronym letter-name expansion".into(),
            version: Some("0.1".into()),
        },
        warnings: vec![PronunciationWarning {
            token: surface.into(),
            kind: PronunciationWarningKind::AcronymExpanded,
            message: format!("acronym expanded: {surface} -> {expanded}"),
        }],
        letter_break_offsets,
        letter_indices,
        part_of_speech: None,
    }
}

fn mixed_alphanumeric_pronunciation(
    word: &WordToken,
    variety: &LinguisticVariety,
) -> WordPronunciation {
    let (candidate, letter_break_offsets, letter_indices) =
        mixed_alphanumeric_sequence(word.text.chars(), variety);
    WordPronunciation {
        candidates: vec![candidate],
        status: PronunciationStatus::Guessed,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: "mixed-alphanumeric pronunciation fallback".into(),
            version: Some("0.1".into()),
        },
        warnings: vec![PronunciationWarning {
            token: word.text.clone(),
            kind: PronunciationWarningKind::MixedAlphaNumeric,
            message: format!("guessed mixed token: {}", word.text),
        }],
        letter_break_offsets,
        letter_indices,
        part_of_speech: None,
    }
}

fn mixed_alphanumeric_sequence(
    characters: impl IntoIterator<Item = char>,
    variety: &LinguisticVariety,
) -> (Vec<PlannedPhoneme>, Vec<usize>, Vec<usize>) {
    let mut candidate = Vec::new();
    let mut break_offsets = Vec::new();
    let mut letter_indices = Vec::new();
    let mut unit_index = 0usize;
    let units = characters
        .into_iter()
        .filter(|character| character.is_alphanumeric())
        .collect::<Vec<_>>();

    for (index, character) in units.iter().enumerate() {
        let pronunciation = if character.is_ascii_digit() {
            orthographic_unit_planned_candidate(
                &character.to_string(),
                variety,
                OrthographicUnitKind::DigitName,
            )
        } else if character.is_alphabetic() {
            orthographic_unit_planned_candidate(
                &character.to_ascii_uppercase().to_string(),
                variety,
                OrthographicUnitKind::LetterName,
            )
        } else {
            Vec::new()
        };
        let letter_index = if character.is_alphabetic() {
            let current = unit_index;
            unit_index += 1;
            current
        } else {
            NO_LETTER_INDEX
        };
        letter_indices.extend(std::iter::repeat_n(letter_index, pronunciation.len()));
        candidate.extend(pronunciation);
        if index + 1 < units.len() {
            break_offsets.push(candidate.len());
        }
    }

    (candidate, break_offsets, letter_indices)
}

fn letter_name_sequence(
    characters: impl IntoIterator<Item = char>,
    variety: &LinguisticVariety,
) -> (Vec<PlannedPhoneme>, Vec<usize>, Vec<usize>) {
    let mut candidate = Vec::new();
    let mut break_offsets = Vec::new();
    let mut letter_indices = Vec::new();
    let letters = characters
        .into_iter()
        .filter(|character| character.is_alphabetic())
        .collect::<Vec<_>>();
    for (index, character) in letters.iter().enumerate() {
        let letter_name = orthographic_unit_planned_candidate(
            &character.to_ascii_uppercase().to_string(),
            variety,
            OrthographicUnitKind::LetterName,
        );
        letter_indices.extend(std::iter::repeat(index).take(letter_name.len()));
        candidate.extend(letter_name);
        if index + 1 < letters.len() {
            break_offsets.push(candidate.len());
        }
    }
    (candidate, break_offsets, letter_indices)
}

fn orthographic_unit_pronunciation(
    word: &WordToken,
    variety: &LinguisticVariety,
    kind: OrthographicUnitKind,
    letter_index: Option<usize>,
) -> WordPronunciation {
    let planned = orthographic_unit_planned_candidate(&word.text, variety, kind);
    let letter_indices = letter_index
        .map(|index| std::iter::repeat_n(index, planned.len()).collect())
        .unwrap_or_default();
    WordPronunciation {
        candidates: vec![planned],
        status: PronunciationStatus::Exact,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: "variety orthographic-unit pronunciation".into(),
            version: Some("0.1".into()),
        },
        warnings: Vec::new(),
        letter_break_offsets: Vec::new(),
        letter_indices,
        part_of_speech: None,
    }
}

fn orthographic_unit_planned_candidate(
    unit: &str,
    variety: &LinguisticVariety,
    kind: OrthographicUnitKind,
) -> Vec<PlannedPhoneme> {
    let normalized = if kind == OrthographicUnitKind::LetterName {
        unit.to_uppercase()
    } else {
        unit.to_string()
    };
    let Some(entry) = variety
        .orthographic_unit_pronunciations
        .iter()
        .find(|entry| entry.kind == kind && entry.unit == normalized)
    else {
        return fallback_orthographic_unit_planned_candidate(&normalized, variety, kind);
    };
    if !entry.source_pronunciation.is_empty() {
        return planned_candidate_from_source_pronunciation(
            variety,
            &entry.source_pronunciation,
            entry.source_pronunciation_notation.as_deref(),
        );
    }
    entry
        .pronunciation
        .iter()
        .filter_map(|id| {
            variety
                .phonemes
                .phonemes
                .get(id)
                .map(planned_phoneme_from_inventory)
        })
        .collect()
}

fn fallback_orthographic_unit_planned_candidate(
    unit: &str,
    variety: &LinguisticVariety,
    kind: OrthographicUnitKind,
) -> Vec<PlannedPhoneme> {
    let context = TokenPronunciationContext {
        next_starts_with_vowelish: false,
        careful_style: true,
        part_of_speech: None,
        next_part_of_speech: None,
    };
    match kind {
        OrthographicUnitKind::DigitName => unit
            .parse::<u32>()
            .ok()
            .and_then(|value| localized_number_word(variety, value, ""))
            .map(|name| planned_candidate_from_orthographic_words(&name, variety, context))
            .unwrap_or_default(),
        OrthographicUnitKind::LetterName => {
            planned_candidate_from_orthography_profile(&unit.to_lowercase(), variety, context)
        }
    }
}

fn planned_candidate_from_orthographic_words(
    text: &str,
    variety: &LinguisticVariety,
    context: TokenPronunciationContext,
) -> Vec<PlannedPhoneme> {
    text.split_whitespace()
        .flat_map(|part| planned_candidate_from_orthography_profile(part, variety, context))
        .collect()
}

fn boundary_phone_token() -> PhoneToken {
    boundary_phone_token_with_id(WORD_BOUNDARY_ID, "word-boundary")
}

fn letter_boundary_phone_token() -> PhoneToken {
    boundary_phone_token_with_id(LETTER_BOUNDARY_ID, "letter-boundary")
}

fn boundary_phone_token_with_id(id: &'static str, method: &'static str) -> PhoneToken {
    PhoneToken {
        phone: Spec::Known(PhoneId::from(id)),
        span: None,
        features: FeatureBundle::default(),
        acoustic_evidence: Vec::new(),
        confidence: 1.0,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: method.into(),
            version: None,
        },
    }
}

fn insert_letter_boundaries(phones: &mut Vec<PhoneToken>, break_offsets: &[usize]) {
    for offset in break_offsets {
        let index = phone_insert_index_for_phoneme_offset(phones, *offset);
        phones.insert(index, letter_boundary_phone_token());
    }
}

fn epenthetic_phones_between_words(
    variety: &LinguisticVariety,
    previous: Option<&PhonemeToken>,
    next: Option<&PhonemeToken>,
) -> Vec<PhoneToken> {
    let (Some(previous), Some(next)) = (previous, next) else {
        return Vec::new();
    };
    epenthetic_phones_after(variety, &[previous.clone(), next.clone()], 0)
}

fn has_pause_boundary_after_word(boundaries: &[SpeechBoundaryToken], word_index: usize) -> bool {
    boundaries.iter().any(|boundary| {
        boundary.after_grapheme_index == word_index
            && (boundary.pause.is_some() || boundary.terminal.is_some())
    })
}

fn realize_connected_allophone_before_word(
    variety: &LinguisticVariety,
    previous_word: Option<&str>,
    phonemes: &mut [PhonemeToken],
    phones: &mut Vec<PhoneToken>,
    next: Option<&PhonemeToken>,
    careful_style: bool,
) {
    let Some(next) = next else {
        return;
    };
    if phonemes.len() < 2 {
        apply_declared_connected_speech(variety, previous_word, phones, next, careful_style);
        return;
    }

    let target_index = phonemes.len() - 1;
    let context = [
        phonemes[target_index - 1].clone(),
        phonemes[target_index].clone(),
        next.clone(),
    ];
    let realized = realize_phoneme_at(
        variety,
        &context,
        1,
        &RealizationOptions {
            careful_style,
            phone_decomposition: PhoneDecompositionPolicy::KeepPhonemic,
            ..Default::default()
        },
    );
    let Some(phone_index) = phones.iter().rposition(|phone| {
        !is_boundary_phone(phone) && !phone.provenance.method.contains("epenthesis rule")
    }) else {
        return;
    };

    phones[phone_index] = realized.clone();
    phonemes[target_index].realized_as = vec![realized];
    apply_declared_connected_speech(variety, previous_word, phones, next, careful_style);
}

fn apply_declared_connected_speech(
    variety: &LinguisticVariety,
    previous_word: Option<&str>,
    phones: &mut Vec<PhoneToken>,
    next: &PhonemeToken,
    careful_style: bool,
) {
    let next_is_vowel = phoneme_token_is_syllabic(variety, next);
    for rule in &variety.connected_speech {
        match rule {
            ConnectedSpeechRule::DeleteFinalPhoneBeforeConsonant { phone } => {
                if !careful_style
                    && !next_is_vowel
                    && final_phone_symbol(phones) == Some(phone.as_str())
                {
                    phones.pop();
                }
            }
            ConnectedSpeechRule::LinkingR { phone } => {
                if next_is_vowel
                    && previous_word.is_some_and(word_has_historical_final_r)
                    && final_phone_symbol(phones) != Some(phone.as_str())
                {
                    phones.push(connected_speech_phone_token(
                        variety,
                        phone,
                        "connected-speech linking r",
                        0.95,
                    ));
                }
            }
            ConnectedSpeechRule::Liaison { entries } => {
                if !next_is_vowel {
                    continue;
                }
                let Some(previous_word) = previous_word else {
                    continue;
                };
                if let Some(entry) = entries
                    .iter()
                    .find(|entry| entry.after_word == previous_word)
                {
                    phones.push(connected_speech_phone_token(
                        variety,
                        &entry.before_vowel_phone,
                        "connected-speech liaison",
                        0.85,
                    ));
                }
            }
        }
    }
}

fn word_has_historical_final_r(word: &str) -> bool {
    let normalized = word
        .trim_matches(|character: char| !character.is_alphabetic())
        .to_lowercase();
    normalized.ends_with('r') || normalized.ends_with("re")
}

fn phoneme_token_is_syllabic(variety: &LinguisticVariety, token: &PhonemeToken) -> bool {
    if token
        .features
        .values
        .get(&FeatureId("phonology.syllabic".into()))
        == Some(&Spec::Known(FeatureValue::Bool(true)))
    {
        return true;
    }
    let Spec::Known(id) = &token.phoneme else {
        return false;
    };
    variety.phonemes.phonemes.get(id).and_then(|phoneme| {
        phoneme
            .features
            .values
            .get(&FeatureId("phonology.syllabic".into()))
    }) == Some(&Spec::Known(FeatureValue::Bool(true)))
}

fn final_phone_symbol(phones: &[PhoneToken]) -> Option<&str> {
    phones.iter().rev().find_map(|phone| {
        if is_boundary_phone(phone) || phone.provenance.method.contains("epenthesis rule") {
            return None;
        }
        let Spec::Known(id) = &phone.phone else {
            return None;
        };
        Some(phone_display_symbol(id))
    })
}

fn connected_speech_phone_token(
    variety: &LinguisticVariety,
    symbol: &str,
    method: &'static str,
    confidence: f32,
) -> PhoneToken {
    let phone_id = PhoneId(format!("ipa.phone.{symbol}").into());
    let features = variety
        .phones
        .phones
        .get(&phone_id)
        .map(|phone| phone.features.clone())
        .unwrap_or_default();
    PhoneToken {
        phone: Spec::Known(phone_id),
        span: None,
        features,
        acoustic_evidence: Vec::new(),
        confidence,
        provenance: EvidenceProvenance {
            source: EvidenceSource::Rule,
            method: method.into(),
            version: Some("0.1".into()),
        },
    }
}

fn phone_insert_index_for_phoneme_offset(phones: &[PhoneToken], offset: usize) -> usize {
    if offset == 0 {
        return 0;
    }

    let mut source_phone_count = 0usize;
    for (index, phone) in phones.iter().enumerate() {
        if is_boundary_phone(phone) || phone.provenance.method.contains("epenthesis rule") {
            continue;
        }
        source_phone_count += 1;
        if source_phone_count == offset {
            return index + 1;
        }
    }

    phones.len()
}

fn assign_realized_phones(phonemes: &mut [PhonemeToken], phones: &[PhoneToken]) {
    let mut phone_iter = phones
        .iter()
        .filter(|phone| !is_boundary_phone(phone))
        .filter(|phone| !phone.provenance.method.contains("epenthesis rule"));
    for phoneme in phonemes {
        if let Some(phone) = phone_iter.next() {
            phoneme.realized_as = vec![phone.clone()];
        }
    }
}

fn is_boundary_phone(phone: &PhoneToken) -> bool {
    matches!(
        &phone.phone,
        Spec::Known(id) if id.as_str().starts_with("boundary.")
    )
}

fn add_letter_index_feature(features: &mut FeatureBundle, letter_index: usize) {
    features.values.insert(
        FeatureId("orthography.letter_index".into()),
        Spec::Known(FeatureValue::Number(letter_index as f64)),
    );
}

fn add_letter_name_feature(features: &mut FeatureBundle) {
    features.values.insert(
        FeatureId("orthography.letter_name".into()),
        Spec::Known(FeatureValue::Bool(true)),
    );
}

fn add_word_index_feature(features: &mut FeatureBundle, word_index: usize) {
    features.values.insert(
        FeatureId("orthography.word_index".into()),
        Spec::Known(FeatureValue::Number(word_index as f64)),
    );
}

fn add_part_of_speech_feature(features: &mut FeatureBundle, part_of_speech: PartOfSpeech) {
    features.values.insert(
        FeatureId("syntax.part_of_speech".into()),
        Spec::Known(FeatureValue::Category(
            part_of_speech_feature_value(part_of_speech).into(),
        )),
    );
}

fn part_of_speech_feature_value(part_of_speech: PartOfSpeech) -> &'static str {
    match part_of_speech {
        PartOfSpeech::Noun => "noun",
        PartOfSpeech::Verb => "verb",
        PartOfSpeech::Auxiliary => "auxiliary",
        PartOfSpeech::Determiner => "determiner",
        PartOfSpeech::Preposition => "preposition",
        PartOfSpeech::Pronoun => "pronoun",
        PartOfSpeech::Adverb => "adverb",
        PartOfSpeech::Adjective => "adjective",
        PartOfSpeech::Conjunction => "conjunction",
        PartOfSpeech::Particle => "particle",
        PartOfSpeech::ProperName => "proper_name",
        PartOfSpeech::Unknown => "unknown",
    }
}

fn confidence_for_status(status: PronunciationStatus) -> f32 {
    match status {
        PronunciationStatus::Exact => 1.0,
        PronunciationStatus::Normalized => 0.95,
        PronunciationStatus::Guessed => 0.55,
        PronunciationStatus::Missing => 0.0,
    }
}

fn status_label(status: PronunciationStatus) -> &'static str {
    match status {
        PronunciationStatus::Exact => "exact",
        PronunciationStatus::Normalized => "normalized",
        PronunciationStatus::Guessed => "guessed",
        PronunciationStatus::Missing => "missing",
    }
}

fn pronunciation_lookup_method(
    lexicon_id: &str,
    status: PronunciationStatus,
    source: &'static str,
    part_of_speech: Option<PartOfSpeech>,
    applied_pos: bool,
) -> String {
    let mut method = if source == lexicon_id {
        format!("{} {} lookup", lexicon_id, status_label(status))
    } else {
        format!("{} {} lookup", source, status_label(status))
    };
    if applied_pos {
        if let Some(part_of_speech) = part_of_speech {
            method = format!(
                "{} + grammar POS {}",
                method,
                part_of_speech_feature_value(part_of_speech)
            );
        }
    }
    method
}

pub fn phoneme_display_symbol(id: &PhonemeId) -> &str {
    id.0.rsplit('.').next().unwrap_or(&id.0)
}

pub fn phoneme_default_phone_display_symbol(id: &PhonemeId, variety: &VarietyId) -> String {
    let variety = variety_by_code(&variety.0).or_else(|| {
        id.0.rsplit_once(".phoneme.")
            .and_then(|(variety_id, _)| variety_by_code(variety_id))
    });
    let Some(variety) = variety else {
        return phoneme_display_symbol(id).to_string();
    };
    let Some(default_phone) = variety
        .phonemes
        .phonemes
        .get(id)
        .and_then(|phoneme| phoneme.default_phone.as_ref())
    else {
        return phoneme_display_symbol(id).to_string();
    };
    phone_display_symbol(default_phone).to_string()
}

pub fn phone_display_symbol(id: &PhoneId) -> &str {
    if matches!(id.as_str(), WORD_BOUNDARY_ID | LETTER_BOUNDARY_ID) {
        return "|";
    }
    id.as_str().rsplit('.').next().unwrap_or(id.as_str())
}

pub fn phoneme_base_symbol(id: &PhonemeId) -> &str {
    let symbol = phoneme_display_symbol(id);
    match symbol.chars().last() {
        Some('0' | '1' | '2') => &symbol[..symbol.len() - 1],
        _ => symbol,
    }
}

fn normalize_text_for_variety(text: &str, variety: &VarietyId) -> String {
    let Some(variety) = variety_by_code(&variety.0) else {
        return text.to_string();
    };
    let mut normalized = apply_spoken_form_rewrites(text, &variety);
    normalized = replace_number_abbreviation_from_variety_data(&normalized, &variety);
    match variety.text_normalization.number_normalization {
        NumberNormalizationProfile::None => normalized,
        NumberNormalizationProfile::SmallNumbers => {
            normalized = normalize_roman_numerals_with_variety(&normalized, &variety);
            normalize_small_numbers_with_variety(&normalized, &variety)
        }
        NumberNormalizationProfile::General => {
            normalized = normalize_roman_numerals_with_variety(&normalized, &variety);
            normalize_general_numbers(&normalized, &variety)
        }
    }
}

fn apply_spoken_form_rewrites(text: &str, variety: &LinguisticVariety) -> String {
    let mut out = text.to_string();
    for rewrite in &variety.text_normalization.spoken_form_rewrites {
        if rewrite.from == "No." {
            continue;
        }
        out = out.replace(&rewrite.from, &rewrite.to);
    }
    out
}

fn replace_number_abbreviation_from_variety_data(
    text: &str,
    variety: &LinguisticVariety,
) -> String {
    let Some(rewrite) = variety
        .text_normalization
        .spoken_form_rewrites
        .iter()
        .find(|rewrite| rewrite.from == "No.")
    else {
        return text.to_string();
    };

    replace_conditional_number_abbreviation(text, &rewrite.from, &rewrite.to)
}

fn normalize_small_numbers_for_variety(text: &str, variety: &VarietyId) -> String {
    let Some(variety) = variety_by_code(&variety.0) else {
        return text.to_string();
    };
    normalize_small_numbers_with_variety(text, &variety)
}

fn normalize_roman_numerals_with_variety(text: &str, variety: &LinguisticVariety) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut index = 0usize;
    let mut previous_word = None::<String>;
    while index < chars.len() {
        if !chars[index].is_alphabetic() {
            out.push(chars[index]);
            index += 1;
            continue;
        }

        let start = index;
        while index < chars.len() && chars[index].is_alphabetic() {
            index += 1;
        }
        let token = chars[start..index].iter().collect::<String>();
        let replacement = roman_numeral_value(&token)
            .filter(|value| should_expand_roman_numeral(&token, *value, previous_word.as_deref()))
            .and_then(|value| spell_cardinal_from_variety(u128::from(value), variety));
        if let Some(word) = replacement {
            out.push_str(&word);
        } else {
            out.push_str(&token);
        }
        previous_word = Some(token.to_lowercase());
    }
    out
}

fn should_expand_roman_numeral(token: &str, value: u32, previous_word: Option<&str>) -> bool {
    if previous_word.is_some_and(is_roman_numeral_context_word) {
        return true;
    }
    token.chars().count() > 1 && value <= 39
}

fn is_roman_numeral_context_word(word: &str) -> bool {
    matches!(
        word,
        "act"
            | "acte"
            | "akt"
            | "book"
            | "buch"
            | "canto"
            | "capitulo"
            | "capítulo"
            | "chapter"
            | "chapitre"
            | "escena"
            | "kapitulo"
            | "liber"
            | "libro"
            | "livre"
            | "part"
            | "parte"
            | "partie"
            | "scene"
            | "scène"
            | "section"
            | "tome"
            | "volume"
    )
}

fn roman_numeral_value(token: &str) -> Option<u32> {
    if token.is_empty()
        || token.chars().count() > 15
        || !token
            .chars()
            .all(|ch| matches!(ch, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
    {
        return None;
    }

    let mut total = 0u32;
    let mut previous = 0u32;
    for ch in token.chars().rev() {
        let value = roman_digit_value(ch)?;
        if value < previous {
            total = total.checked_sub(value)?;
        } else {
            total = total.checked_add(value)?;
            previous = value;
        }
    }

    if total == 0 || total > 3999 {
        return None;
    }
    (canonical_roman_numeral(total) == token).then_some(total)
}

fn roman_digit_value(ch: char) -> Option<u32> {
    Some(match ch {
        'I' => 1,
        'V' => 5,
        'X' => 10,
        'L' => 50,
        'C' => 100,
        'D' => 500,
        'M' => 1000,
        _ => return None,
    })
}

fn canonical_roman_numeral(mut value: u32) -> String {
    let mut out = String::new();
    for (arabic, roman) in [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ] {
        while value >= arabic {
            out.push_str(roman);
            value -= arabic;
        }
    }
    out
}

fn normalize_small_numbers_with_variety(text: &str, variety: &LinguisticVariety) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut index = 0usize;
    while index < chars.len() {
        if !chars[index].is_ascii_digit() {
            out.push(chars[index]);
            index += 1;
            continue;
        }

        let start = index;
        while index < chars.len() && chars[index].is_ascii_digit() {
            index += 1;
        }
        let digits = chars[start..index].iter().collect::<String>();
        let suffix_start = index;
        while index < chars.len() && chars[index].is_alphabetic() {
            index += 1;
        }
        let suffix = chars[suffix_start..index].iter().collect::<String>();
        if start > 0 && chars[start - 1].is_alphabetic() {
            out.push_str(&digits);
            out.push_str(&suffix);
            continue;
        }
        let replacement = digits
            .parse::<u32>()
            .ok()
            .and_then(|value| localized_number_word(variety, value, &suffix));
        if let Some(word) = replacement {
            out.push_str(&word);
        } else {
            out.push_str(&digits);
            out.push_str(&suffix);
        }
    }
    out
}

fn localized_number_word(variety: &LinguisticVariety, value: u32, suffix: &str) -> Option<String> {
    let names = variety.number_names.as_ref()?;
    if !suffix.is_empty() {
        if let Some(ordinal) = names.ordinal_suffixes.iter().find(|ordinal| {
            ordinal.value == value && ordinal.suffixes.iter().any(|candidate| candidate == suffix)
        }) {
            return Some(ordinal.name.clone());
        }
        return None;
    }
    names.cardinal_0_to_20.get(value as usize).cloned()
}

fn spell_out(value: u128, variety: &LinguisticVariety) -> String {
    spell_cardinal_from_variety(value, variety).unwrap_or_else(|| value.to_string())
}

fn spell_cardinal_from_variety(value: u128, variety: &LinguisticVariety) -> Option<String> {
    let names = variety.number_names.as_ref()?;
    if let Some(name) = names.cardinal_0_to_20.get(value as usize) {
        return Some(name.clone());
    }
    if let Some(name) = names
        .cardinal_tens
        .iter()
        .find(|name| u128::from(name.value) == value)
        .map(|name| name.name.clone())
    {
        return Some(name);
    }
    if let Some(name) = names
        .special_number_names
        .iter()
        .find(|name| u128::from(name.value) == value)
        .map(|name| name.name.clone())
    {
        return Some(name);
    }
    match value {
        21..=99 => {
            let head = value - (value % 10);
            let tail = value % 10;
            Some(format!(
                "{}-{}",
                spell_cardinal_from_variety(head, variety)?,
                spell_cardinal_from_variety(tail, variety)?
            ))
        }
        100..=999 => {
            let head = value / 100;
            let tail = value % 100;
            let hundred = format!(
                "{} {}",
                spell_cardinal_from_variety(head, variety)?,
                names.hundred_name.as_deref()?
            );
            if tail > 0 {
                Some(format!(
                    "{} {}",
                    hundred,
                    spell_cardinal_from_variety(tail, variety)?
                ))
            } else {
                Some(hundred)
            }
        }
        _ => {
            let scale = names
                .scale_names
                .iter()
                .filter(|scale| u128::from(scale.power) <= digit_count_power(value))
                .max_by_key(|scale| scale.power)?;
            let divisor = 10u128.pow(scale.power);
            let head = value / divisor;
            let tail = value % divisor;
            let head = spell_cardinal_from_variety(head, variety)?;
            if tail > 0 {
                Some(format!(
                    "{} {} {}",
                    head,
                    scale.name,
                    spell_cardinal_from_variety(tail, variety)?
                ))
            } else {
                Some(format!("{} {}", head, scale.name))
            }
        }
    }
}

fn spell_ordinal_from_variety(value: u128, variety: &LinguisticVariety) -> Option<String> {
    let names = variety.number_names.as_ref()?;
    if let Some(name) = names
        .ordinal_names
        .iter()
        .find(|name| u128::from(name.value) == value)
        .map(|name| name.name.clone())
    {
        return Some(name);
    }
    if value < 100 && value % 10 != 0 {
        return Some(format!(
            "{}-{}",
            spell_cardinal_from_variety(value - (value % 10), variety)?,
            spell_ordinal_from_variety(value % 10, variety)?
        ));
    }
    None
}

fn digit_count_power(mut value: u128) -> u128 {
    let mut power = 0;
    while value >= 10 {
        value /= 10;
        power += 1;
    }
    power
}

fn decimal_separator_name(variety: &LinguisticVariety) -> Option<&str> {
    variety
        .number_names
        .as_ref()
        .and_then(|names| names.decimal_separator_name.as_deref())
}

fn range_separator_name(variety: &LinguisticVariety) -> Option<&str> {
    variety
        .number_names
        .as_ref()
        .and_then(|names| names.range_separator_name.as_deref())
}

fn product_separator_name(variety: &LinguisticVariety) -> Option<&str> {
    variety
        .number_names
        .as_ref()
        .and_then(|names| names.product_separator_name.as_deref())
}

fn slash_separator_name(variety: &LinguisticVariety) -> Option<&str> {
    variety
        .number_names
        .as_ref()
        .and_then(|names| names.slash_separator_name.as_deref())
}

fn date_separator_name(variety: &LinguisticVariety) -> Option<&str> {
    variety
        .number_names
        .as_ref()
        .and_then(|names| names.date_separator_name.as_deref())
}

fn currency_major_singular(variety: &LinguisticVariety) -> Option<&str> {
    variety
        .number_names
        .as_ref()
        .and_then(|names| names.currency_major_singular.as_deref())
}

fn currency_major_plural(variety: &LinguisticVariety) -> Option<&str> {
    variety
        .number_names
        .as_ref()
        .and_then(|names| names.currency_major_plural.as_deref())
}

fn currency_minor_singular(variety: &LinguisticVariety) -> Option<&str> {
    variety
        .number_names
        .as_ref()
        .and_then(|names| names.currency_minor_singular.as_deref())
}

fn currency_minor_plural(variety: &LinguisticVariety) -> Option<&str> {
    variety
        .number_names
        .as_ref()
        .and_then(|names| names.currency_minor_plural.as_deref())
}

fn special_number_name(variety: &LinguisticVariety, value: u128) -> Option<String> {
    variety
        .number_names
        .as_ref()?
        .special_number_names
        .iter()
        .find(|name| u128::from(name.value) == value)
        .map(|name| name.name.clone())
}

fn spell_digit_sequence(digits: &str, variety: &LinguisticVariety) -> String {
    digits
        .chars()
        .filter_map(|digit| digit.to_digit(10))
        .map(|digit| spell_out(digit as u128, variety))
        .collect::<Vec<_>>()
        .join(" ")
}

fn spell_clock_time(
    hour: u128,
    minute_digits: &str,
    variety: &LinguisticVariety,
) -> Option<String> {
    if minute_digits.len() != 2 || !minute_digits.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let minutes = minute_digits.parse::<u128>().ok()?;
    if minutes > 59 {
        return None;
    }
    let names = variety.number_names.as_ref()?;
    let minute_text = if minutes == 0 {
        names.clock_zero_minute_name.clone()?
    } else if minutes < 10 {
        format!(
            "{} {}",
            names.clock_leading_zero_name.as_deref()?,
            spell_out(minutes, variety)
        )
    } else {
        spell_out(minutes, variety)
    };
    Some(format!("{} {}", spell_out(hour, variety), minute_text))
}

fn spell_dotted_number(
    first: u128,
    segments: &[String],
    variety: &LinguisticVariety,
) -> Option<String> {
    let point = decimal_separator_name(variety)?;
    let mut parts = vec![spell_out(first, variety)];
    for segment in segments {
        parts.push(point.to_string());
        parts.push(spell_digit_sequence(segment, variety));
    }
    Some(parts.join(" "))
}

fn spell_year(year: u128, variety: &LinguisticVariety) -> String {
    if (2000..=2009).contains(&year) {
        format!(
            "{} {}",
            spell_out(2000, variety),
            spell_out(year - 2000, variety)
        )
    } else if (2010..=2099).contains(&year) {
        format!(
            "{} {}",
            spell_out(20, variety),
            spell_out(year - 2000, variety)
        )
    } else {
        spell_out(year, variety)
    }
}

fn spell_ordinal(value: u128, variety: &LinguisticVariety) -> String {
    spell_ordinal_from_variety(value, variety).unwrap_or_else(|| value.to_string())
}

fn ordinal_at(
    characters: &[char],
    start: usize,
    variety: &LinguisticVariety,
) -> Option<(String, usize)> {
    let mut index = start;
    let mut digits = String::new();
    while index < characters.len() && characters[index].is_ascii_digit() {
        digits.push(characters[index]);
        index += 1;
    }
    if digits.is_empty() || index + 1 >= characters.len() {
        return None;
    }
    let value = digits.parse::<u128>().ok()?;
    let suffixes = &variety.number_names.as_ref()?.ordinal_suffixes;
    let Some(end) = suffixes
        .iter()
        .filter(|ordinal| u128::from(ordinal.value) == value)
        .flat_map(|ordinal| ordinal.suffixes.iter())
        .find_map(|suffix| {
            let suffix_len = suffix.chars().count();
            let end = index + suffix_len;
            (end <= characters.len()
                && characters[index..end]
                    .iter()
                    .collect::<String>()
                    .eq_ignore_ascii_case(suffix))
            .then_some(end)
        })
    else {
        return None;
    };
    if end < characters.len() && characters[end].is_alphanumeric() {
        return None;
    }
    Some((spell_ordinal(value, variety), end))
}

fn leading_decimal_at(
    characters: &[char],
    start: usize,
    variety: &LinguisticVariety,
) -> Option<(String, usize)> {
    if characters.get(start).copied() != Some('.') {
        return None;
    }
    let mut index = start + 1;
    let mut digits = String::new();
    while index < characters.len() && characters[index].is_ascii_digit() {
        digits.push(characters[index]);
        index += 1;
    }
    if digits.is_empty() {
        return None;
    }

    let mut suffix_temp = index;
    while suffix_temp < characters.len() && characters[suffix_temp] == ' ' {
        suffix_temp += 1;
    }
    let mut suffix_word = String::new();
    while suffix_temp < characters.len() && characters[suffix_temp].is_alphabetic() {
        suffix_word.push(characters[suffix_temp]);
        suffix_temp += 1;
    }
    if !suffix_word.is_empty()
        && is_known_unit(variety, &suffix_word.to_lowercase())
        && (suffix_temp >= characters.len() || !characters[suffix_temp].is_alphanumeric())
    {
        return Some((
            format!(
                "{} {} {}",
                decimal_separator_name(variety)?,
                spell_digit_sequence(&digits, variety),
                spell_unit(variety, 2, &suffix_word.to_lowercase()).unwrap_or_default()
            ),
            suffix_temp,
        ));
    }

    Some((
        format!(
            "{} {}",
            decimal_separator_name(variety)?,
            spell_digit_sequence(&digits, variety)
        ),
        index,
    ))
}

fn numeric_product_at(
    characters: &[char],
    start: usize,
    variety: &LinguisticVariety,
) -> Option<(String, usize)> {
    let mut left = String::new();
    let mut index = start;
    while index < characters.len() && characters[index].is_ascii_digit() {
        left.push(characters[index]);
        index += 1;
    }
    if left.is_empty() || index >= characters.len() || !matches!(characters[index], 'x' | 'X' | '×')
    {
        return None;
    }

    index += 1;
    let mut right = String::new();
    while index < characters.len() && characters[index].is_ascii_digit() {
        right.push(characters[index]);
        index += 1;
    }
    if right.is_empty() || (index < characters.len() && characters[index].is_alphanumeric()) {
        return None;
    }

    let left = left.parse::<u128>().ok()?;
    let right = right.parse::<u128>().ok()?;
    Some((
        format!(
            "{} {} {}",
            spell_out(left, variety),
            product_separator_name(variety)?,
            spell_out(right, variety)
        ),
        index,
    ))
}

fn quoted_height_at(
    characters: &[char],
    start: usize,
    variety: &LinguisticVariety,
) -> Option<(String, usize)> {
    let mut feet = String::new();
    let mut index = start;
    while index < characters.len() && characters[index].is_ascii_digit() {
        feet.push(characters[index]);
        index += 1;
    }
    if feet.is_empty() || characters.get(index).copied() != Some('\'') {
        return None;
    }
    index += 1;
    let mut inches = String::new();
    while index < characters.len() && characters[index].is_ascii_digit() {
        inches.push(characters[index]);
        index += 1;
    }
    if inches.is_empty() {
        return None;
    }
    if matches!(characters.get(index), Some('"') | Some('”')) {
        index += 1;
    }
    let feet = feet.parse::<u128>().ok()?;
    let inches = inches.parse::<u128>().ok()?;
    Some((
        format!(
            "{} {} {}",
            spell_out(feet, variety),
            spell_unit(variety, 1, "foot").unwrap_or_else(|| "foot".into()),
            spell_out(inches, variety)
        ),
        index,
    ))
}

fn dashed_number_at(
    characters: &[char],
    start: usize,
    variety: &LinguisticVariety,
) -> Option<(String, usize)> {
    let mut groups = Vec::new();
    let mut index = start;
    loop {
        let group_start = index;
        while index < characters.len() && characters[index].is_ascii_digit() {
            index += 1;
        }
        if group_start == index {
            return None;
        }
        groups.push(characters[group_start..index].iter().collect::<String>());
        if index >= characters.len() || characters[index] != '-' {
            break;
        }
        index += 1;
    }
    if groups.len() < 2 || (index < characters.len() && characters[index].is_ascii_digit()) {
        return None;
    }
    let mut suffix_temp = index;
    while suffix_temp < characters.len() && characters[suffix_temp] == ' ' {
        suffix_temp += 1;
    }
    let mut suffix_word = String::new();
    while suffix_temp < characters.len() && characters[suffix_temp].is_alphabetic() {
        suffix_word.push(characters[suffix_temp]);
        suffix_temp += 1;
    }
    let suffix = if !suffix_word.is_empty()
        && is_known_unit(variety, &suffix_word.to_lowercase())
        && (suffix_temp >= characters.len() || !characters[suffix_temp].is_alphanumeric())
    {
        spell_unit(variety, 2, &suffix_word.to_lowercase()).map(|unit| (unit, suffix_temp))
    } else {
        None
    };

    if groups
        .iter()
        .map(String::len)
        .collect::<Vec<_>>()
        .as_slice()
        == [4, 2, 2]
    {
        let year = groups[0].parse::<u128>().ok()?;
        let day = groups[2].parse::<u128>().ok()?;
        let day_spelled = if groups[2].starts_with('0') {
            spell_digit_sequence(&groups[2], variety)
        } else {
            spell_out(day, variety)
        };
        let parts = [
            spell_year(year, variety),
            spell_digit_sequence(&groups[1], variety),
            day_spelled,
        ];
        let date_separator = date_separator_name(variety)?;
        return Some((parts.join(&format!(" {} ", date_separator)), index));
    }
    let range_separator = range_separator_name(variety)?;
    let parts = groups
        .iter()
        .map(|group| Some(spell_out(group.parse::<u128>().ok()?, variety)))
        .collect::<Option<Vec<_>>>()?;
    let mut text = parts.join(&format!(" {} ", range_separator));
    let end = if let Some((unit, suffix_end)) = suffix {
        text.push(' ');
        text.push_str(&unit);
        suffix_end
    } else {
        index
    };
    Some((text, end))
}

fn slash_number_at(
    characters: &[char],
    start: usize,
    variety: &LinguisticVariety,
) -> Option<(String, usize)> {
    let mut groups = Vec::new();
    let mut index = start;
    loop {
        let group_start = index;
        while index < characters.len() && characters[index].is_ascii_digit() {
            index += 1;
        }
        if group_start == index {
            return None;
        }
        groups.push(characters[group_start..index].iter().collect::<String>());
        if index >= characters.len() || characters[index] != '/' {
            break;
        }
        index += 1;
    }

    if groups.len() < 2 || (index < characters.len() && characters[index].is_alphanumeric()) {
        return None;
    }

    let slash_separator = slash_separator_name(variety)?;
    let parts = groups
        .iter()
        .enumerate()
        .map(|(group_index, group)| {
            let value = group.parse::<u128>().ok()?;
            if group_index + 1 == groups.len() && group.len() == 4 {
                Some(spell_year(value, variety))
            } else {
                Some(spell_out(value, variety))
            }
        })
        .collect::<Option<Vec<_>>>()?;
    Some((parts.join(&format!(" {} ", slash_separator)), index))
}

fn phone_number_digits_at(characters: &[char], start: usize) -> Option<(String, usize)> {
    let mut groups = Vec::new();
    let mut index = start;
    loop {
        let group_start = index;
        while index < characters.len() && characters[index].is_ascii_digit() {
            index += 1;
        }
        if group_start == index {
            return None;
        }
        groups.push(characters[group_start..index].iter().collect::<String>());
        if index >= characters.len() || characters[index] != '-' {
            break;
        }
        index += 1;
    }

    let shape_is_phone = matches!(
        groups
            .iter()
            .map(String::len)
            .collect::<Vec<_>>()
            .as_slice(),
        [3, 4] | [3, 3, 4]
    );
    if !shape_is_phone {
        return None;
    }
    if index < characters.len() && characters[index].is_alphanumeric() {
        return None;
    }

    Some((groups.concat(), index))
}

fn is_scale_word(variety: &LinguisticVariety, word: &str) -> bool {
    variety
        .number_names
        .as_ref()
        .is_some_and(|names| names.scale_names.iter().any(|scale| scale.name == word))
}

fn is_known_unit(variety: &LinguisticVariety, word: &str) -> bool {
    variety.number_names.as_ref().is_some_and(|names| {
        names
            .unit_names
            .iter()
            .any(|unit| unit.aliases.iter().any(|alias| alias == word))
    })
}

fn spell_unit(variety: &LinguisticVariety, val: u128, unit_alias: &str) -> Option<String> {
    let unit = variety
        .number_names
        .as_ref()?
        .unit_names
        .iter()
        .find(|unit| unit.aliases.iter().any(|alias| alias == unit_alias))?;
    Some(if val == 1 {
        unit.singular.clone()
    } else {
        unit.plural.clone()
    })
}

fn is_number_or_scale_word(variety: &LinguisticVariety, word: &str) -> bool {
    if is_scale_word(variety, word) {
        return true;
    }
    word.chars()
        .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
}

fn is_linked_as_modifier(
    word_idx: usize,
    syntax: &crate::syntax::SentenceSyntaxAnalysis,
    words: &[WordToken],
    variety: &LinguisticVariety,
) -> bool {
    let parse = match syntax.primary_parse() {
        Some(p) => p,
        None => return false,
    };

    let mut current_idx = word_idx;
    let mut visited = std::collections::HashSet::new();
    visited.insert(current_idx);

    while let Some(link) = parse.links.iter().find(|l| {
        l.left == current_idx
            && (l.kind == crate::syntax::SyntacticLinkKind::NounCompound
                || l.kind == crate::syntax::SyntacticLinkKind::Modifier)
            && !visited.contains(&l.right)
    }) {
        current_idx = link.right;
        visited.insert(current_idx);

        if let Some(target_word) = words.get(current_idx) {
            let text_lower = target_word.normalized.to_lowercase();
            if !is_number_or_scale_word(variety, &text_lower) {
                return true;
            }
        }
    }

    false
}

fn find_word_index_at(c_idx: usize, words: &[WordToken]) -> Option<usize> {
    words
        .iter()
        .position(|w| w.span.start_char <= c_idx && c_idx < w.span.end_char)
}

pub fn normalize_general_numbers(text: &str, variety: &LinguisticVariety) -> String {
    let words = tokenize_words(text);
    let words_str: Vec<String> = words.iter().map(|w| w.text.clone()).collect();
    let syntax = if let Some(analyzer) = variety.syntax_analyzer {
        analyzer(&words_str, None)
    } else {
        VarietyGrammarParser::new(variety.id.clone()).parse(&words_str, None)
    };

    let char_vec: Vec<char> = text.chars().collect();
    let mut result = String::new();
    let mut idx = 0;

    while idx < char_vec.len() {
        // Check currency amount at `idx`
        if char_vec[idx] == '$' {
            let mut int_part = String::new();
            let mut temp_idx = idx + 1;
            while temp_idx < char_vec.len()
                && (char_vec[temp_idx].is_ascii_digit()
                    || (char_vec[temp_idx] == ','
                        && char_vec
                            .get(temp_idx + 1)
                            .is_some_and(|ch| ch.is_ascii_digit())))
            {
                int_part.push(char_vec[temp_idx]);
                temp_idx += 1;
            }

            let mut cents_part: Option<String> = None;
            let mut cents_temp = temp_idx;
            if cents_temp < char_vec.len() && char_vec[cents_temp] == '.' {
                cents_temp += 1;
                if cents_temp + 1 < char_vec.len()
                    && char_vec[cents_temp].is_ascii_digit()
                    && char_vec[cents_temp + 1].is_ascii_digit()
                {
                    if cents_temp + 2 >= char_vec.len()
                        || !char_vec[cents_temp + 2].is_ascii_digit()
                    {
                        cents_part = Some(format!(
                            "{}{}",
                            char_vec[cents_temp],
                            char_vec[cents_temp + 1]
                        ));
                        temp_idx = cents_temp + 2;
                    }
                }
            }

            let mut scale_temp = temp_idx;
            while scale_temp < char_vec.len() && char_vec[scale_temp] == ' ' {
                scale_temp += 1;
            }
            let mut scale_word = String::new();
            while scale_temp < char_vec.len() && char_vec[scale_temp].is_alphabetic() {
                scale_word.push(char_vec[scale_temp]);
                scale_temp += 1;
            }

            let scale_valid = !scale_word.is_empty()
                && is_scale_word(variety, &scale_word.to_lowercase())
                && (scale_temp >= char_vec.len() || !char_vec[scale_temp].is_alphanumeric());

            let actual_scale = if scale_valid {
                temp_idx = scale_temp;
                Some(scale_word.to_lowercase())
            } else {
                None
            };

            let clean_int: String = int_part.chars().filter(|&c| c != ',').collect();
            let commas_valid = if int_part.contains(',') {
                let groups: Vec<&str> = int_part.split(',').collect();
                groups.first().is_some_and(|g| {
                    !g.is_empty() && g.len() <= 3 && g.chars().all(|c| c.is_ascii_digit())
                }) && groups
                    .iter()
                    .skip(1)
                    .all(|g| g.len() == 3 && g.chars().all(|c| c.is_ascii_digit()))
            } else {
                true
            };

            if !clean_int.is_empty() && commas_valid {
                if let Ok(dollars_val) = clean_int.parse::<u128>() {
                    let word_idx = find_word_index_at(idx + 1, &words).unwrap_or(0);
                    let query_idx = if actual_scale.is_some() {
                        find_word_index_at(temp_idx - 1, &words).unwrap_or(word_idx)
                    } else {
                        word_idx
                    };

                    let modifier = is_linked_as_modifier(query_idx, &syntax, &words, variety);

                    let spelled = if modifier {
                        let Some(dollar_unit) = currency_major_singular(variety) else {
                            result.push_str(&text[idx..temp_idx]);
                            idx = temp_idx;
                            continue;
                        };
                        let dollars_spelled = spell_out(dollars_val, variety);
                        let base = if let Some(ref scale) = actual_scale {
                            format!("{}-{}", dollars_spelled, scale)
                        } else {
                            dollars_spelled
                        };
                        if let Some(ref cents_str) = cents_part {
                            if let Ok(cents_val) = cents_str.parse::<u128>() {
                                let Some(cent_unit) = currency_minor_singular(variety) else {
                                    result.push_str(&text[idx..temp_idx]);
                                    idx = temp_idx;
                                    continue;
                                };
                                let cents_spelled = spell_out(cents_val, variety);
                                format!(
                                    "{}-{}-and-{}-{}",
                                    base.replace(' ', "-"),
                                    dollar_unit,
                                    cents_spelled.replace(' ', "-"),
                                    cent_unit
                                )
                            } else {
                                format!("{}-{}", base.replace(' ', "-"), dollar_unit)
                            }
                        } else {
                            format!("{}-{}", base.replace(' ', "-"), dollar_unit)
                        }
                    } else {
                        let dollars_spelled = spell_out(dollars_val, variety);
                        let base = if let Some(ref scale) = actual_scale {
                            format!("{} {}", dollars_spelled, scale)
                        } else {
                            dollars_spelled
                        };
                        let dollars_unit = if dollars_val == 1 && actual_scale.is_none() {
                            currency_major_singular(variety)
                        } else {
                            currency_major_plural(variety)
                        };
                        let Some(dollars_unit) = dollars_unit else {
                            result.push_str(&text[idx..temp_idx]);
                            idx = temp_idx;
                            continue;
                        };
                        if let Some(ref cents_str) = cents_part {
                            if let Ok(cents_val) = cents_str.parse::<u128>() {
                                let cents_spelled = spell_out(cents_val, variety);
                                let cents_unit = if cents_val == 1 {
                                    currency_minor_singular(variety)
                                } else {
                                    currency_minor_plural(variety)
                                };
                                let Some(cents_unit) = cents_unit else {
                                    result.push_str(&text[idx..temp_idx]);
                                    idx = temp_idx;
                                    continue;
                                };
                                if dollars_val == 0 {
                                    format!("{} {}", cents_spelled, cents_unit)
                                } else if cents_val == 0 {
                                    format!("{} {}", base, dollars_unit)
                                } else {
                                    format!(
                                        "{} {} and {} {}",
                                        base, dollars_unit, cents_spelled, cents_unit
                                    )
                                }
                            } else {
                                format!("{} {}", base, dollars_unit)
                            }
                        } else {
                            format!("{} {}", base, dollars_unit)
                        }
                    };

                    result.push_str(&spelled);
                    idx = temp_idx;
                    continue;
                }
            }
        }

        if let Some((decimal, decimal_end)) = leading_decimal_at(&char_vec, idx, variety) {
            result.push_str(&decimal);
            idx = decimal_end;
            continue;
        }

        // Check cardinal number or measurement at `idx`
        if char_vec[idx].is_ascii_digit() {
            if let Some((ordinal, ordinal_end)) = ordinal_at(&char_vec, idx, variety) {
                result.push_str(&ordinal);
                idx = ordinal_end;
                continue;
            }
            if let Some((height, height_end)) = quoted_height_at(&char_vec, idx, variety) {
                result.push_str(&height);
                idx = height_end;
                continue;
            }
            if let Some((product, product_end)) = numeric_product_at(&char_vec, idx, variety) {
                result.push_str(&product);
                idx = product_end;
                continue;
            }
            if let Some((slash_number, slash_end)) = slash_number_at(&char_vec, idx, variety) {
                result.push_str(&slash_number);
                idx = slash_end;
                continue;
            }

            let mut int_part = String::new();
            let mut temp_idx = idx;
            while temp_idx < char_vec.len()
                && (char_vec[temp_idx].is_ascii_digit()
                    || (char_vec[temp_idx] == ','
                        && char_vec
                            .get(temp_idx + 1)
                            .is_some_and(|ch| ch.is_ascii_digit())))
            {
                int_part.push(char_vec[temp_idx]);
                temp_idx += 1;
            }

            // Check if this digit sequence is part of a mixed alphanumeric word
            let mut is_part_of_word = false;
            if idx > 0 && char_vec[idx - 1].is_alphabetic() {
                is_part_of_word = true;
            }
            if !is_part_of_word && temp_idx < char_vec.len() && char_vec[temp_idx].is_alphabetic() {
                let mut suffix_temp = temp_idx;
                let mut suffix_word = String::new();
                while suffix_temp < char_vec.len() && char_vec[suffix_temp].is_alphabetic() {
                    suffix_word.push(char_vec[suffix_temp]);
                    suffix_temp += 1;
                }
                let suffix_valid = !suffix_word.is_empty()
                    && is_known_unit(variety, &suffix_word.to_lowercase())
                    && (suffix_temp >= char_vec.len() || !char_vec[suffix_temp].is_alphanumeric());
                if !suffix_valid {
                    is_part_of_word = true;
                }
            }

            if is_part_of_word {
                result.push_str(&int_part);
                idx = temp_idx;
                continue;
            }

            if let Some((phone_digits, phone_end)) = phone_number_digits_at(&char_vec, idx) {
                result.push_str(&spell_digit_sequence(&phone_digits, variety));
                idx = phone_end;
                continue;
            }

            let clean_int: String = int_part.chars().filter(|&c| c != ',').collect();
            if let Some(special) = clean_int
                .parse::<u128>()
                .ok()
                .and_then(|value| special_number_name(variety, value))
                && (temp_idx >= char_vec.len() || !char_vec[temp_idx].is_alphanumeric())
            {
                result.push_str(&special);
                idx = temp_idx;
                continue;
            }
            if let Some((dashed_number, dashed_end)) = dashed_number_at(&char_vec, idx, variety) {
                result.push_str(&dashed_number);
                idx = dashed_end;
                continue;
            }
            let commas_valid = if int_part.contains(',') {
                let groups: Vec<&str> = int_part.split(',').collect();
                groups.first().is_some_and(|g| {
                    !g.is_empty() && g.len() <= 3 && g.chars().all(|c| c.is_ascii_digit())
                }) && groups
                    .iter()
                    .skip(1)
                    .all(|g| g.len() == 3 && g.chars().all(|c| c.is_ascii_digit()))
            } else {
                true
            };

            if !clean_int.is_empty() && commas_valid {
                if let Ok(val) = clean_int.parse::<u128>() {
                    if temp_idx < char_vec.len() && char_vec[temp_idx] == ':' {
                        let minute_start = temp_idx + 1;
                        let minute_end = minute_start.saturating_add(2);
                        if minute_end <= char_vec.len()
                            && char_vec[minute_start..minute_end]
                                .iter()
                                .all(|ch| ch.is_ascii_digit())
                            && (minute_end >= char_vec.len()
                                || !char_vec[minute_end].is_ascii_digit())
                        {
                            let minute_digits = char_vec[minute_start..minute_end]
                                .iter()
                                .collect::<String>();
                            if let Some(spelled) = spell_clock_time(val, &minute_digits, variety) {
                                result.push_str(&spelled);
                                idx = minute_end;
                                continue;
                            }
                        }
                    }

                    let mut suffix_temp = temp_idx;
                    while suffix_temp < char_vec.len() && char_vec[suffix_temp] == ' ' {
                        suffix_temp += 1;
                    }

                    let mut suffix_word = String::new();
                    if suffix_temp < char_vec.len() && char_vec[suffix_temp] == '%' {
                        suffix_word.push('%');
                        suffix_temp += 1;
                    } else {
                        while suffix_temp < char_vec.len() && char_vec[suffix_temp].is_alphabetic()
                        {
                            suffix_word.push(char_vec[suffix_temp]);
                            suffix_temp += 1;
                        }
                    }

                    let suffix_valid = !suffix_word.is_empty()
                        && is_known_unit(variety, &suffix_word.to_lowercase())
                        && (suffix_temp >= char_vec.len()
                            || !char_vec[suffix_temp].is_alphanumeric());

                    if suffix_valid {
                        let unit_str = suffix_word.to_lowercase();
                        let mut height_inches_val: Option<u128> = None;
                        let mut height_temp = suffix_temp;

                        if matches!(unit_str.as_str(), "ft" | "feet" | "foot") {
                            while height_temp < char_vec.len() && char_vec[height_temp] == ' ' {
                                height_temp += 1;
                            }
                            let mut inches_str = String::new();
                            while height_temp < char_vec.len()
                                && char_vec[height_temp].is_ascii_digit()
                            {
                                inches_str.push(char_vec[height_temp]);
                                height_temp += 1;
                            }
                            let inches_valid = !inches_str.is_empty()
                                && (height_temp >= char_vec.len()
                                    || !char_vec[height_temp].is_alphanumeric());

                            if inches_valid {
                                if let Ok(inches) = inches_str.parse::<u128>() {
                                    height_inches_val = Some(inches);
                                }
                            }
                        }

                        let spelled = if let Some(inches) = height_inches_val {
                            temp_idx = height_temp;
                            format!(
                                "{} {} {}",
                                spell_out(val, variety),
                                spell_unit(variety, 1, "foot").unwrap_or_else(|| "foot".into()),
                                spell_out(inches, variety)
                            )
                        } else {
                            temp_idx = suffix_temp;
                            let unit_spelled =
                                spell_unit(variety, val, &unit_str).unwrap_or_default();
                            format!("{} {}", spell_out(val, variety), unit_spelled)
                        };

                        result.push_str(&spelled);
                        idx = temp_idx;
                        continue;
                    } else {
                        let mut decimal_temp = temp_idx;
                        if decimal_temp < char_vec.len() && char_vec[decimal_temp] == '.' {
                            let mut segments = Vec::new();
                            while decimal_temp < char_vec.len() && char_vec[decimal_temp] == '.' {
                                let segment_start = decimal_temp + 1;
                                let mut segment_temp = segment_start;
                                let mut segment = String::new();
                                while segment_temp < char_vec.len()
                                    && char_vec[segment_temp].is_ascii_digit()
                                {
                                    segment.push(char_vec[segment_temp]);
                                    segment_temp += 1;
                                }
                                if segment.is_empty() {
                                    break;
                                }
                                decimal_temp = segment_temp;
                                segments.push(segment);
                            }
                            if !segments.is_empty() {
                                let mut decimal_suffix_temp = decimal_temp;
                                while decimal_suffix_temp < char_vec.len()
                                    && char_vec[decimal_suffix_temp] == ' '
                                {
                                    decimal_suffix_temp += 1;
                                }

                                let mut decimal_suffix_word = String::new();
                                while decimal_suffix_temp < char_vec.len()
                                    && char_vec[decimal_suffix_temp].is_alphabetic()
                                {
                                    decimal_suffix_word.push(char_vec[decimal_suffix_temp]);
                                    decimal_suffix_temp += 1;
                                }
                                let decimal_suffix_valid = !decimal_suffix_word.is_empty()
                                    && is_known_unit(variety, &decimal_suffix_word.to_lowercase())
                                    && (decimal_suffix_temp >= char_vec.len()
                                        || !char_vec[decimal_suffix_temp].is_alphanumeric());

                                if decimal_suffix_valid {
                                    let unit_str = decimal_suffix_word.to_lowercase();
                                    if let Some(dotted) =
                                        spell_dotted_number(val, &segments, variety)
                                    {
                                        result.push_str(&format!(
                                            "{} {}",
                                            dotted,
                                            spell_unit(variety, val, &unit_str).unwrap_or_default()
                                        ));
                                        idx = decimal_suffix_temp;
                                        continue;
                                    }
                                }

                                if let Some(dotted) = spell_dotted_number(val, &segments, variety) {
                                    result.push_str(&dotted);
                                    idx = decimal_temp;
                                    continue;
                                }
                            }
                        }

                        let spelled = spell_out(val, variety);
                        result.push_str(&spelled);
                        idx = temp_idx;
                        continue;
                    }
                }
            }
        }

        result.push(char_vec[idx]);
        idx += 1;
    }

    result
}

pub fn spoken_form_for_variety(text: &str, variety: &VarietyId) -> String {
    let Some(variety) = variety_by_code(&variety.0) else {
        return text.to_string();
    };
    let out = apply_spoken_form_rewrites(text, &variety);
    replace_number_abbreviation_from_variety_data(&out, &variety)
}

#[cfg(test)]
fn english_normalize_numbers(text: &str) -> String {
    let variety = variety_by_code("en-US").expect("built-in English variety");
    normalize_general_numbers(text, &variety)
}

fn replace_conditional_number_abbreviation(text: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(index) = rest.find(from) {
        out.push_str(&rest[..index]);
        let after = &rest[index + from.len()..];
        if after
            .trim_start()
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_digit())
        {
            out.push_str(to);
        } else {
            out.push_str(from);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::builtin_varieties;
    use crate::rules::RuleCondition;
    use crate::syntax::SyntacticLinkKind;
    use crate::variety::VarietyImplementationStatus;

    #[test]
    fn test_number_normalization() {
        assert_eq!(
            english_normalize_numbers("He has $15 million."),
            "He has fifteen million dollars."
        );
        assert_eq!(
            english_normalize_numbers("A $15 million renovation."),
            "A fifteen-million-dollar renovation."
        );
        assert_eq!(
            english_normalize_numbers("His $15 million slush fund."),
            "His fifteen-million-dollar slush fund."
        );
        assert_eq!(
            english_normalize_numbers("He is 6ft 4."),
            "He is six foot four."
        );
        assert_eq!(
            english_normalize_numbers("We reached 60mph."),
            "We reached sixty miles per hour."
        );
        assert_eq!(
            english_normalize_numbers("The price was $15.52."),
            "The price was fifteen dollars and fifty-two cents."
        );
        assert_eq!(
            english_normalize_numbers("I saw 3.14 written on the board."),
            "I saw three point one four written on the board."
        );
        assert_eq!(
            english_normalize_numbers("Prof. Adams arrived at 4:30 p.m. sharp."),
            "Prof. Adams arrived at four thirty p.m. sharp."
        );
        assert_eq!(
            english_normalize_numbers("The version is 1.2.3."),
            "The version is one point two point three."
        );
        assert_eq!(
            english_normalize_numbers("Call 555-1212 now."),
            "Call five five five one two one two now."
        );
        assert_eq!(
            english_normalize_numbers("The dose was 2.0 mg."),
            "The dose was two point zero milligrams."
        );
        assert_eq!(
            english_normalize_numbers("Take .5 mg daily."),
            "Take point five milligrams daily."
        );
        assert_eq!(
            english_normalize_numbers("He is 6'4\"."),
            "He is six foot four."
        );
        assert_eq!(english_normalize_numbers("2x4"), "two by four");
        assert_eq!(
            english_normalize_numbers("Pages 3-5 are missing."),
            "Pages three to five are missing."
        );
        assert_eq!(
            english_normalize_numbers("The range is 10-12 mg."),
            "The range is ten to twelve milligrams."
        );
        assert_eq!(
            english_normalize_numbers("5/6/2026"),
            "five slash six slash twenty twenty-six"
        );
        assert_eq!(
            english_normalize_numbers("2026-06-16"),
            "twenty twenty-six dash zero six dash sixteen"
        );
        assert_eq!(
            english_normalize_numbers("$12.50"),
            "twelve dollars and fifty cents"
        );
        assert_eq!(english_normalize_numbers("$0.99"), "ninety-nine cents");
        assert_eq!(english_normalize_numbers("$1.00"), "one dollar");
        assert_eq!(
            english_normalize_numbers("Call 911 now."),
            "Call nine one one now."
        );
        assert_eq!(
            english_normalize_numbers("She finished 1st."),
            "She finished first."
        );
        assert_eq!(english_normalize_numbers("He came 2nd."), "He came second.");
        assert_eq!(
            english_normalize_numbers("They ranked 3rd."),
            "They ranked third."
        );
        assert_eq!(
            english_normalize_numbers("This is the 21st case."),
            "This is the twenty-first case."
        );
        assert_eq!(
            VarietyDataPhonemicizer
                .text_normalizer("It was 70°F outside.", &VarietyId("en-US".into())),
            "It was seventy degrees Fahrenheit outside."
        );
        assert_eq!(
            VarietyDataPhonemicizer
                .text_normalizer("The CPU ran at 3.5GHz.", &VarietyId("en-US".into())),
            "The CPU ran at three point five gigahertz."
        );
        assert_eq!(
            VarietyDataPhonemicizer.text_normalizer("No. 5", &VarietyId("en-US".into())),
            "Number five"
        );
        assert_eq!(
            english_normalize_numbers("A 5% discount."),
            "A five percent discount."
        );
        assert_eq!(
            english_normalize_numbers("affects over 100,000 pending immigration cases"),
            "affects over one hundred thousand pending immigration cases"
        );
    }

    #[test]
    fn english_spoken_form_expands_shared_pronunciation_rewrites() {
        assert_eq!(
            spoken_form_for_variety(
                "Dr. Smith saw No. 5 at 3.14 p.m.",
                &VarietyId("en-US".into())
            ),
            "Doctor Smith saw Number 5 at 3.14 p m"
        );
        assert_eq!(
            spoken_form_for_variety("The Loadstone Rock", &VarietyId("en-US".into())),
            "The Lodestone Rock"
        );
        assert_eq!(
            spoken_form_for_variety(
                "He lives on Sansome St. The house is blue.",
                &VarietyId("en-US".into())
            ),
            "He lives on Sansome St. The house is blue."
        );
    }

    #[test]
    fn test_tts_pronunciation_pipeline_regression() {
        let phonemicizer = VarietyDataPhonemicizer;

        // Test "logorrhea" (override)
        let out_logo = phonemicizer
            .phonemicize(&request("logorrhea", "en-US"))
            .unwrap();
        let syms_logo = cmudict_symbols(&out_logo);
        assert_eq!(syms_logo, vec!["L", "AO2", "G", "ER0", "IY1", "AH0"]);

        // Test "talkativeness" (morphology)
        let out_talk = phonemicizer
            .phonemicize(&request("talkativeness", "en-US"))
            .unwrap();
        let syms_talk = cmudict_symbols(&out_talk);
        assert_eq!(
            syms_talk,
            vec!["T", "AO1", "K", "AH0", "T", "IH0", "V", "N", "AH0", "S"]
        );

        // Test "wordiness" (morphology)
        let out_word = phonemicizer
            .phonemicize(&request("wordiness", "en-US"))
            .unwrap();
        let syms_word = cmudict_symbols(&out_word);
        assert_eq!(syms_word, vec!["W", "ER1", "D", "IY0", "N", "AH0", "S"]);

        // Test "excessive" (base dict/morphology)
        let out_exc = phonemicizer
            .phonemicize(&request("excessive", "en-US"))
            .unwrap();
        let syms_exc = cmudict_symbols(&out_exc);
        assert_eq!(syms_exc, vec!["IH0", "K", "S", "EH1", "S", "IH0", "V"]);

        // Test "incoherent" (base dict/morphology)
        let out_inc = phonemicizer
            .phonemicize(&request("incoherent", "en-US"))
            .unwrap();
        let syms_inc = cmudict_symbols(&out_inc);
        assert_eq!(
            syms_inc,
            vec!["IH2", "N", "K", "OW0", "HH", "IH1", "R", "AH0", "N", "T"]
        );

        // Test fallback (humble G2P with single stress)
        let out_fallback = phonemicizer
            .phonemicize(&request("xyzzyqux", "en-US"))
            .unwrap();
        let syms_fallback = cmudict_symbols(&out_fallback);
        let primary_stress_count = syms_fallback.iter().filter(|s| s.ends_with('1')).count();
        assert!(
            primary_stress_count <= 1,
            "Fallback should never have more than one primary stress"
        );
    }

    fn request(text: &str, variety: &str) -> PhonemicizeRequest {
        PhonemicizeRequest {
            text: text.into(),
            variety: VarietyId(variety.into()),
            style: None,
        }
    }

    fn phoneme_symbols(output: &PhonemicizeOutput) -> Vec<String> {
        output
            .phonemes
            .iter()
            .filter_map(|token| match &token.phoneme {
                Spec::Known(id) => Some(phoneme_display_symbol(id).to_string()),
                _ => None,
            })
            .collect()
    }

    fn cmudict_symbols(output: &PhonemicizeOutput) -> Vec<String> {
        output
            .phonemes
            .iter()
            .filter_map(|token| {
                let base = phoneme_feature_category(token, "phonology.base_symbol")?;
                let stress = phoneme_feature_category(token, "phonology.stress")
                    .and_then(cmu_stress_digit)
                    .unwrap_or_default();
                Some(format!("{base}{stress}"))
            })
            .collect()
    }

    fn cmudict_symbols_for_word(output: &PhonemicizeOutput, word_index: usize) -> Vec<String> {
        output
            .phonemes
            .iter()
            .filter(|token| {
                phoneme_usize_feature(token, "orthography.word_index") == Some(word_index)
            })
            .filter_map(|token| {
                let base = phoneme_feature_category(token, "phonology.base_symbol")?;
                let stress = phoneme_feature_category(token, "phonology.stress")
                    .and_then(cmu_stress_digit)
                    .unwrap_or_default();
                Some(format!("{base}{stress}"))
            })
            .collect()
    }

    fn phoneme_feature_category<'a>(token: &'a PhonemeToken, feature_id: &str) -> Option<&'a str> {
        let value = token.features.values.get(&FeatureId(feature_id.into()))?;
        match value {
            Spec::Known(FeatureValue::Category(value)) | Spec::Known(FeatureValue::Text(value)) => {
                Some(value.as_str())
            }
            _ => None,
        }
    }

    fn phoneme_usize_feature(token: &PhonemeToken, feature_id: &str) -> Option<usize> {
        let value = token.features.values.get(&FeatureId(feature_id.into()))?;
        match value {
            Spec::Known(FeatureValue::Number(value)) if value.is_finite() && *value >= 0.0 => {
                Some(*value as usize)
            }
            _ => None,
        }
    }

    fn phone_feature_category<'a>(token: &'a PhoneToken, feature_id: &str) -> Option<&'a str> {
        let value = token.features.values.get(&FeatureId(feature_id.into()))?;
        match value {
            Spec::Known(FeatureValue::Category(value)) | Spec::Known(FeatureValue::Text(value)) => {
                Some(value.as_str())
            }
            _ => None,
        }
    }

    fn phone_feature_bool(token: &PhoneToken, feature_id: &str) -> Option<bool> {
        let value = token.features.values.get(&FeatureId(feature_id.into()))?;
        match value {
            Spec::Known(FeatureValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    fn cmu_stress_digit(stress: &str) -> Option<&'static str> {
        match stress {
            "unstressed" => Some("0"),
            "primary" => Some("1"),
            "secondary" => Some("2"),
            _ => None,
        }
    }

    fn phone_symbols(output: &PhonemicizeOutput) -> Vec<String> {
        output
            .phones
            .iter()
            .filter_map(|token| match &token.phone {
                Spec::Known(id) => Some(phone_display_symbol(id).to_string()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn interpreter_training_contract_has_word_indices_and_realized_phones() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request(
                "Mr. Carter can't email Dr. Smith at 4:30 p.m.",
                "en-US",
            ))
            .expect("training-style transcript should phonemicize");

        assert_eq!(
            output
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            [
                "Mister", "Carter", "can't", "email", "Doctor", "Smith", "at", "four", "thirty",
                "p", "m"
            ]
        );
        assert!(output.phonemes.iter().all(|token| {
            phoneme_usize_feature(token, "orthography.word_index").is_some()
                && token.realized_as.len() == 1
                && token
                    .realized_as
                    .iter()
                    .all(|phone| !is_boundary_phone(phone))
        }));
        assert!(output.warnings.iter().all(|warning| {
            !matches!(
                warning.kind,
                PronunciationWarningKind::GuessedWord
                    | PronunciationWarningKind::MixedAlphaNumeric
                    | PronunciationWarningKind::UnknownPronunciation
            )
        }));
        assert!(
            output
                .boundaries
                .iter()
                .any(|boundary| boundary.terminal == Some(TerminalPunctuation::Period))
        );
    }

    #[test]
    fn digit_and_initialism_expansion_preserves_training_alignment_metadata() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("U.S. officials met Apollo 11.", "en-US"))
            .expect("initialism and number should phonemicize");

        assert_eq!(
            output
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            ["U", "S", "officials", "met", "Apollo", "eleven"]
        );
        assert_eq!(cmudict_symbols_for_word(&output, 0), ["Y", "UW1"]);
        assert_eq!(cmudict_symbols_for_word(&output, 1), ["EH1", "S"]);
        assert_eq!(
            cmudict_symbols_for_word(&output, 5),
            ["IH0", "L", "EH1", "V", "AH0", "N"]
        );
        assert!(
            output.phonemes.iter().all(|token| {
                phoneme_usize_feature(token, "orthography.word_index")
                    .is_some_and(|index| index < output.graphemes.len())
            }),
            "every phoneme label should map back to a grapheme token"
        );
    }

    #[test]
    fn all_caps_transcripts_use_dataset_pronunciations_before_acronym_fallback() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request(
                "AND CLASSIFIED BY SCIENCE WHICH IS TO SUFFERING",
                "en-US",
            ))
            .expect("all-caps transcript should phonemicize");

        assert_eq!(
            output
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            [
                "AND",
                "CLASSIFIED",
                "BY",
                "SCIENCE",
                "WHICH",
                "IS",
                "TO",
                "SUFFERING"
            ]
        );
        assert!(
            output
                .warnings
                .iter()
                .all(|warning| warning.kind != PronunciationWarningKind::AcronymExpanded),
            "ordinary all-caps transcript words should not be spelled as acronyms: {:?}",
            output.warnings
        );
        assert_eq!(
            cmudict_symbols_for_word(&output, 1),
            ["K", "L", "AE1", "S", "AH0", "F", "AY2", "D"]
        );
        assert_eq!(cmudict_symbols_for_word(&output, 2), ["B", "AY1"]);
        assert_eq!(cmudict_symbols_for_word(&output, 4), ["W", "IH1", "CH"]);
    }

    #[test]
    fn all_caps_unknown_names_are_not_poisoned_as_acronyms() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("MARIUS HAD BUT A STEP MORE TO TAKE", "en-US"))
            .expect("all-caps transcript with unknown name should phonemicize");

        assert!(
            output
                .warnings
                .iter()
                .all(|warning| warning.kind != PronunciationWarningKind::AcronymExpanded),
            "dataset casing should not force letter-name expansion: {:?}",
            output.warnings
        );
        assert_ne!(
            cmudict_symbols_for_word(&output, 0),
            ["EH1", "M", "EY1", "AA1", "R", "AY1", "Y", "UW1", "EH1", "S"]
        );
        assert_eq!(cmudict_symbols_for_word(&output, 1), ["HH", "AE1", "D"]);
        assert_eq!(cmudict_symbols_for_word(&output, 4), ["S", "T", "EH1", "P"]);
        assert_eq!(cmudict_symbols_for_word(&output, 5), ["M", "AO1", "R"]);
        assert_eq!(cmudict_symbols_for_word(&output, 7), ["T", "EY1", "K"]);
    }

    #[test]
    fn unknown_dataset_names_use_grapheme_clusters_not_raw_letters() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("CORINTHE PONTMERCY CHANVRERIE MONDETOUR", "en-US"))
            .expect("unknown dataset names should be reported");

        assert!(
            output.phonemes.is_empty(),
            "unknown dataset names should not use an English grapheme guesser"
        );
        assert_eq!(
            output
                .warnings
                .iter()
                .filter(|warning| warning.kind == PronunciationWarningKind::UnknownPronunciation)
                .count(),
            4
        );
    }

    #[test]
    fn regular_inflections_do_not_use_hidden_english_lemma_fallback() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("PARCELLED COMBATED ANNIHILATES", "en-US"))
            .expect("regular inflections should be reported when undeclared");

        assert!(output.phonemes.is_empty());
        assert_eq!(
            output
                .warnings
                .iter()
                .filter(|warning| warning.kind == PronunciationWarningKind::UnknownPronunciation)
                .count(),
            3
        );
    }

    #[test]
    fn possessive_names_should_not_fall_back_to_letter_pronunciation() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("Alice's email arrived.", "en-US"))
            .expect("possessive name should phonemicize");

        assert!(
            output
                .warnings
                .iter()
                .all(|warning| warning.kind != PronunciationWarningKind::GuessedWord)
        );
        assert_eq!(
            cmudict_symbols_for_word(&output, 0),
            ["AE1", "L", "AH0", "S", "AH0", "Z"]
        );
    }

    #[test]
    fn pronounceable_acronyms_should_not_always_be_initialisms() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("NASA launched Apollo 11.", "en-US"))
            .expect("acronym sentence should phonemicize");

        assert_eq!(
            cmudict_symbols_for_word(&output, 0),
            ["N", "AE1", "S", "AH0"]
        );
        assert!(
            output
                .warnings
                .iter()
                .all(|warning| warning.kind != PronunciationWarningKind::AcronymExpanded)
        );
    }

    #[test]
    #[ignore = "known gap: productive hyphenated prefixes are split into separate words"]
    fn hyphenated_prefixed_words_should_compose_before_word_splitting() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("The co-op re-opened.", "en-US"))
            .expect("hyphenated prefixed words should phonemicize");

        assert_eq!(
            output
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            ["The", "co-op", "re-opened"]
        );
        assert_eq!(
            cmudict_symbols_for_word(&output, 2),
            ["R", "IY0", "OW1", "P", "AH0", "N", "D"]
        );
    }

    #[test]
    fn hyphenated_compounds_compose_as_one_word_for_discrepancy_reports() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("just-in-time", "en-US"))
            .expect("hyphenated compound should phonemicize");

        assert_eq!(
            output
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            ["just-in-time"]
        );
        assert_eq!(
            cmudict_symbols_for_word(&output, 0),
            ["JH", "AH1", "S", "T", "IH0", "N", "T", "AY1", "M"]
        );
        assert!(
            output
                .warnings
                .iter()
                .all(|warning| warning.kind != PronunciationWarningKind::UnknownPronunciation)
        );
    }

    #[test]
    fn grammar_pos_disambiguates_cmudict_heteronyms() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("I record the permit.", "en-US"))
            .expect("heteronyms should phonemicize");

        assert_eq!(output.syntax.tokens[1].pos, PartOfSpeech::Verb);
        assert_eq!(output.syntax.tokens[3].pos, PartOfSpeech::Noun);
        assert_eq!(
            cmudict_symbols_for_word(&output, 1),
            ["R", "AH0", "K", "AO1", "R", "D"]
        );
        assert_eq!(
            cmudict_symbols_for_word(&output, 3),
            ["P", "ER1", "M", "IH2", "T"]
        );
        assert_eq!(
            phoneme_feature_category(&output.phonemes[1], "syntax.part_of_speech"),
            Some("verb")
        );
        assert!(
            output.phonemes[1]
                .provenance
                .method
                .contains("grammar POS verb")
        );
    }

    #[test]
    fn grammar_pos_can_select_noun_then_verb_for_same_spelling() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("The object will object.", "en-US"))
            .expect("heteronyms should phonemicize");

        assert_eq!(output.syntax.tokens[1].pos, PartOfSpeech::Noun);
        assert_eq!(output.syntax.tokens[3].pos, PartOfSpeech::Verb);
        assert_eq!(
            cmudict_symbols_for_word(&output, 1),
            ["AA1", "B", "JH", "EH0", "K", "T"]
        );
        assert_eq!(
            cmudict_symbols_for_word(&output, 3),
            ["AH0", "B", "JH", "EH1", "K", "T"]
        );
    }

    #[test]
    fn hello_world_uses_cmudict_not_characters() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("hello world", "en-US"))
            .expect("en-US should phonemicize");

        assert_eq!(
            phoneme_symbols(&output),
            ["h", "ʌ", "l", "oʊ", "w", "ɝ", "l", "d"]
        );
        assert_eq!(
            cmudict_symbols(&output),
            ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"]
        );
        assert_ne!(phoneme_symbols(&output), ["h", "e", "l", "l", "o"]);
        assert!(
            output
                .phonemes
                .iter()
                .all(|token| token.provenance.source == EvidenceSource::Lexicon)
        );
    }

    #[test]
    fn acceptance_words_match_cmudict_expectations() {
        for (word, expected) in [
            ("doctor", vec!["D", "AA1", "K", "T", "ER0"]),
            (
                "fitzgerald",
                vec!["F", "IH0", "T", "S", "JH", "EH1", "R", "AH0", "L", "D"],
            ),
            ("xylophone", vec!["Z", "AY1", "L", "AH0", "F", "OW2", "N"]),
            ("okay", vec!["OW2", "K", "EY1"]),
        ] {
            let output = VarietyDataPhonemicizer
                .phonemicize(&request(word, "en-US-GA"))
                .expect("word should phonemicize");
            assert_eq!(cmudict_symbols(&output), expected, "{word}");
        }
    }

    #[test]
    fn curly_apostrophe_contractions_use_cmudict_entry() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("I’ll", "en-US"))
            .expect("contraction should phonemicize");

        assert_eq!(phoneme_symbols(&output), ["aɪ", "l"]);
        assert_eq!(cmudict_symbols(&output), ["AY1", "L"]);
        assert!(output.warnings.iter().all(|warning| {
            !matches!(
                warning.kind,
                PronunciationWarningKind::GuessedWord
                    | PronunciationWarningKind::MixedAlphaNumeric
                    | PronunciationWarningKind::UnknownPronunciation
            )
        }));
        assert!(
            output
                .phonemes
                .iter()
                .all(|token| token.provenance.source == EvidenceSource::Lexicon)
        );
    }

    #[test]
    fn hyphenated_mixed_tokens_split_before_fallback() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("speech-to-StyleTTS2", "en-US"))
            .expect("mixed token should phonemicize");

        assert_eq!(
            cmudict_symbols(&output),
            [
                "S", "P", "IY1", "CH", "T", "AH0", "S", "T", "AY1", "L", "T", "IY1", "T", "IY1",
                "EH1", "S", "T", "UW1"
            ]
        );
        assert_eq!(
            output
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            ["speech", "to", "Style", "T", "T", "S", "2"]
        );
        assert!(
            output
                .warnings
                .iter()
                .all(|warning| warning.kind != PronunciationWarningKind::MixedAlphaNumeric)
        );
    }

    #[test]
    fn weak_forms_and_unstressed_ah_realize_as_schwa() {
        let the_cat = VarietyDataPhonemicizer
            .phonemicize(&request("the cat", "en-US"))
            .expect("the cat");
        assert_eq!(&phone_symbols(&the_cat)[..2], ["ð", "ə"]);
        assert!(!phone_symbols(&the_cat)[..2].contains(&"ʌ".into()));
        assert!(the_cat.warnings.is_empty());
        assert!(
            the_cat.phonemes[0]
                .provenance
                .method
                .contains("the before consonant")
        );

        let the_apple = VarietyDataPhonemicizer
            .phonemicize(&request("the apple", "en-US"))
            .expect("the apple");
        assert_eq!(&phoneme_symbols(&the_apple)[..2], ["ð", "iː"]);
        assert_eq!(&cmudict_symbols(&the_apple)[..2], ["DH", "IY0"]);
        assert_eq!(&phone_symbols(&the_apple)[..2], ["ð", "iː"]);

        let and_then = VarietyDataPhonemicizer
            .phonemicize(&request("and then", "en-US"))
            .expect("and then");
        assert_eq!(&phone_symbols(&and_then)[..3], ["ə", "n", "d"]);
    }

    #[test]
    fn cmudict_unstressed_vowels_reduce_without_changing_stressed_strut() {
        let current = VarietyDataPhonemicizer
            .phonemicize(&request("current", "en-US"))
            .expect("current");
        assert_eq!(phone_symbols(&current), ["kʰ", "ɝ", "ə", "n", "t"]);

        let termination = VarietyDataPhonemicizer
            .phonemicize(&request("termination", "en-US"))
            .expect("termination");
        assert_eq!(
            phone_symbols(&termination),
            ["t", "ɚ", "m", "ə", "n", "eɪ", "ʃ", "ə", "n"]
        );

        let preserves = VarietyDataPhonemicizer
            .phonemicize(&request("preserves", "en-US"))
            .expect("preserves");
        assert_eq!(&phone_symbols(&preserves)[..3], ["p", "ɹ", "ə"]);

        let strut = VarietyDataPhonemicizer
            .phonemicize(&request("strut", "en-US"))
            .expect("strut");
        assert!(phone_symbols(&strut).contains(&"ʌ".into()));
    }

    #[test]
    fn acronyms_expand_as_letter_names_and_mixed_tokens_warn() {
        let ir = VarietyDataPhonemicizer
            .phonemicize(&request("Use IR", "en-US"))
            .expect("IR");
        assert_eq!(cmudict_symbols_for_word(&ir, 1), ["AY1", "AA1", "R"]);
        assert!(ir.warnings.iter().any(|warning| {
            warning.kind == PronunciationWarningKind::AcronymExpanded && warning.token == "IR"
        }));

        let spaced_ir = VarietyDataPhonemicizer
            .phonemicize(&request("I R", "en-US"))
            .expect("spaced IR");
        assert_eq!(phoneme_symbols(&spaced_ir), ["aɪ", "ɑ", "ɹ"]);
        assert_eq!(cmudict_symbols(&spaced_ir), ["AY1", "AA1", "R"]);
        assert_eq!(phone_symbols(&spaced_ir), ["aɪ", "|", "j", "ɑ", "ɹ"]);

        let paused_ir = VarietyDataPhonemicizer
            .phonemicize(&request("I, R", "en-US"))
            .expect("paused IR");
        assert_eq!(phoneme_symbols(&paused_ir), ["aɪ", "ɑ", "ɹ"]);
        assert_eq!(cmudict_symbols(&paused_ir), ["AY1", "AA1", "R"]);
        assert_eq!(phone_symbols(&paused_ir), ["aɪ", "|", "ɑ", "ɹ"]);

        let styletts2 = VarietyDataPhonemicizer
            .phonemicize(&request("StyleTTS2", "en-US"))
            .expect("StyleTTS2");
        assert_eq!(
            styletts2
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            ["Style", "T", "T", "S", "2"]
        );
        assert!(
            styletts2
                .warnings
                .iter()
                .all(|warning| warning.kind != PronunciationWarningKind::MixedAlphaNumeric)
        );
    }

    #[test]
    fn dotted_initials_are_letter_names_not_weak_articles() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("A. B. Carter signed the note.", "en-US"))
            .expect("initials should phonemicize");
        let symbols = cmudict_symbols(&output);
        assert_eq!(&symbols[..3], &["EY1", "B", "IY1"]);
    }

    #[test]
    fn requested_edge_cases_phonemicize_without_obvious_normalization_errors() {
        let dr = VarietyDataPhonemicizer
            .phonemicize(&request("Dr. Smith", "en-US"))
            .expect("doctor abbreviation");
        assert_eq!(
            &cmudict_symbols(&dr)[..6],
            &["D", "AA1", "K", "T", "ER0", "S"]
        );

        let saint = VarietyDataPhonemicizer
            .phonemicize(&request("St. John went home.", "en-US"))
            .expect("saint abbreviation");
        assert!(
            cmudict_symbols(&saint)
                .windows(4)
                .any(|window| window == ["S", "EY1", "N", "T"])
        );

        let street = VarietyDataPhonemicizer
            .phonemicize(&request("He lives on Sansome St.", "en-US"))
            .expect("street abbreviation");
        assert!(
            cmudict_symbols(&street)
                .windows(5)
                .any(|window| window == ["S", "T", "R", "IY1", "T"])
        );

        let lead_noun = VarietyDataPhonemicizer
            .phonemicize(&request("The lead pipe broke.", "en-US"))
            .expect("lead noun");
        assert!(
            cmudict_symbols(&lead_noun)
                .windows(3)
                .any(|window| window == ["L", "EH1", "D"])
        );

        let lead_verb = VarietyDataPhonemicizer
            .phonemicize(&request("They lead the team.", "en-US"))
            .expect("lead verb");
        assert!(
            cmudict_symbols(&lead_verb)
                .windows(3)
                .any(|window| window == ["L", "IY1", "D"])
        );

        let read_past = VarietyDataPhonemicizer
            .phonemicize(&request("I read the book yesterday.", "en-US"))
            .expect("read ambiguous");
        let read_present = VarietyDataPhonemicizer
            .phonemicize(&request("I read the book today.", "en-US"))
            .expect("read ambiguous");
        assert!(
            cmudict_symbols(&read_past)
                .windows(3)
                .any(|window| window == ["R", "EH1", "D"])
        );
        assert!(
            cmudict_symbols(&read_present)
                .windows(3)
                .any(|window| window == ["R", "EH1", "D"]),
            "read/read tense remains ambiguous without stronger syntax"
        );

        assert_eq!(
            VarietyDataPhonemicizer.text_normalizer("AT&T called.", &VarietyId("en-US".into())),
            "A T and T called."
        );
        assert_eq!(
            VarietyDataPhonemicizer.text_normalizer("R&D approved it.", &VarietyId("en-US".into())),
            "R and D approved it."
        );
        assert_eq!(
            VarietyDataPhonemicizer
                .text_normalizer("C++ is different from C#.", &VarietyId("en-US".into())),
            "C plus plus is different from C sharp."
        );
    }

    #[test]
    fn water_flaps_in_ga_and_careful_style_blocks_it() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("water", "en-US-GA"))
            .expect("water");
        assert!(phone_symbols(&output).contains(&"ɾ".into()));
        let flapped_t = output
            .phonemes
            .iter()
            .find(|token| {
                matches!(
                    &token.phoneme,
                    Spec::Known(id) if phoneme_display_symbol(id) == "t"
                )
            })
            .expect("T phoneme");
        assert_eq!(
            flapped_t
                .realized_as
                .iter()
                .filter_map(|phone| match &phone.phone {
                    Spec::Known(id) => Some(phone_display_symbol(id).to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            ["ɾ"]
        );

        let careful = VarietyDataPhonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: "water".into(),
                variety: VarietyId("en-US-GA".into()),
                style: Some(PhonemicizeStyle {
                    careful_style: true,
                    ..PhonemicizeStyle::default()
                }),
            })
            .expect("water careful");
        assert!(phone_symbols(&careful).contains(&"t".into()));
        assert!(!phone_symbols(&careful).contains(&"ɾ".into()));
    }

    #[test]
    fn flapping_can_apply_across_unpaused_word_boundaries() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("not a", "en-US-GA"))
            .expect("not a");
        assert_eq!(phone_symbols(&output), ["n", "ɑ", "ɾ", "|", "ə"]);

        let flapped_t = output
            .phonemes
            .iter()
            .find(|token| {
                matches!(
                    &token.phoneme,
                    Spec::Known(id) if phoneme_display_symbol(id) == "t"
                )
            })
            .expect("T phoneme");
        assert_eq!(
            flapped_t
                .realized_as
                .iter()
                .filter_map(|phone| match &phone.phone {
                    Spec::Known(id) => Some(phone_display_symbol(id).to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            ["ɾ"]
        );

        let paused = VarietyDataPhonemicizer
            .phonemicize(&request("not, a", "en-US-GA"))
            .expect("not, a");
        assert_eq!(phone_symbols(&paused), ["n", "ɑ", "t", "|", "ə"]);
    }

    #[test]
    fn nasal_assimilation_applies_only_before_velars() {
        let before_k = VarietyDataPhonemicizer
            .phonemicize(&request("in case", "en-US"))
            .expect("declared words before velar");
        assert!(phone_symbols(&before_k).contains(&"ŋ".into()));

        let before_d = VarietyDataPhonemicizer
            .phonemicize(&request("in debt", "en-US"))
            .expect("declared words before non-velar");
        assert!(phone_symbols(&before_d).contains(&"n".into()));
        assert!(!phone_symbols(&before_d).contains(&"ŋ".into()));
    }

    #[test]
    fn final_devoicing_marks_final_z_without_rewriting_phone() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("seas", "en-US"))
            .expect("seas should phonemicize");
        let final_phone = output
            .phones
            .iter()
            .rev()
            .find(|phone| !is_boundary_phone(phone))
            .expect("final speech phone");

        assert!(matches!(
            &final_phone.phone,
            Spec::Known(id) if id.as_str() == "ipa.phone.z"
        ));
        assert_eq!(
            phone_feature_category(final_phone, "phonology.voicing"),
            Some("voiced")
        );
        assert_eq!(
            phone_feature_bool(final_phone, "phonology.partial_devoicing"),
            Some(true)
        );
        assert_eq!(
            phone_feature_category(final_phone, "phonology.devoicing"),
            Some("final_optional")
        );
    }

    #[test]
    fn final_devoicing_does_not_mark_nonfinal_initial_z() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("zoo", "en-US"))
            .expect("zoo should phonemicize");
        let initial_phone = output
            .phones
            .iter()
            .find(|phone| !is_boundary_phone(phone))
            .expect("initial speech phone");

        assert!(matches!(
            &initial_phone.phone,
            Spec::Known(id) if id.as_str() == "ipa.phone.z"
        ));
        assert_eq!(
            phone_feature_category(initial_phone, "phonology.voicing"),
            Some("voiced")
        );
        assert_ne!(
            phone_feature_bool(initial_phone, "phonology.partial_devoicing"),
            Some(true)
        );
    }

    #[test]
    fn aliases_and_implementation_status_are_data_driven() {
        let en_us = VarietyDataPhonemicizer
            .phonemicize(&request("okay", "en-US"))
            .expect("en-US alias");
        let ga = VarietyDataPhonemicizer
            .phonemicize(&request("okay", "en-US-GA"))
            .expect("GA");
        assert_eq!(phoneme_symbols(&en_us), phoneme_symbols(&ga));

        let rp = variety_by_code("en-GB.RP").expect("RP alias");
        assert_eq!(
            rp.implementation_status,
            VarietyImplementationStatus::Complete
        );
        assert_eq!(rp.id.0, "en-GB-RP");
    }

    #[test]
    fn rp_uses_its_own_vowels_non_rhoticity_and_yod() {
        for (word, expected_phonemes, expected_phones) in [
            ("lot", vec!["ɒ"], vec!["l", "ɒ", "t"]),
            ("bath", vec!["ɑː"], vec!["b", "ɑː", "θ"]),
            ("goat", vec!["əʊ"], vec!["ɡ", "əʊ", "t"]),
            ("nurse", vec!["ɜː"], vec!["n", "ɜː", "s"]),
            ("near", vec!["ɪə"], vec!["n", "ɪə"]),
            ("water", vec!["ɔː", "ə"], vec!["w", "ɔː", "t", "ə"]),
            ("tune", vec!["uː"], vec!["t", "j", "uː", "n"]),
        ] {
            let output = VarietyDataPhonemicizer
                .phonemicize(&request(word, "en-GB-RP"))
                .unwrap_or_else(|error| panic!("{word} should phonemicize: {error}"));
            let phonemes = phoneme_symbols(&output);
            for expected in expected_phonemes {
                assert!(phonemes.iter().any(|symbol| symbol == expected), "{word}");
            }
            assert_eq!(phone_symbols(&output), expected_phones, "{word}");
        }
    }

    #[test]
    fn rp_linking_r_only_surfaces_before_a_following_vowel() {
        let before_vowel = VarietyDataPhonemicizer
            .phonemicize(&request("far away", "en-GB-RP"))
            .expect("linking r phrase");
        let before_consonant = VarietyDataPhonemicizer
            .phonemicize(&request("far north", "en-GB-RP"))
            .expect("non-rhotic phrase");

        assert_eq!(
            phone_symbols(&before_vowel)
                .into_iter()
                .filter(|symbol| symbol != "|")
                .collect::<Vec<_>>(),
            ["f", "ɑː", "ɹ", "ə", "w", "eɪ"]
        );
        assert_eq!(
            phone_symbols(&before_consonant)
                .into_iter()
                .filter(|symbol| symbol != "|")
                .collect::<Vec<_>>(),
            ["f", "ɑː", "n", "ɔː", "θ"]
        );
    }

    #[test]
    fn rp_uses_british_letter_names_and_non_rhotic_four() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("R Z 4", "en-GB-RP"))
            .expect("RP orthographic units");

        assert_eq!(
            phone_symbols(&output)
                .into_iter()
                .filter(|symbol| symbol != "|")
                .collect::<Vec<_>>(),
            ["ɑː", "z", "e", "d", "f", "ɔː"]
        );
    }

    #[test]
    fn french_uses_lexique_before_rule_fallback() {
        let phonemicizer =
            phonemicizer_for_variety(&VarietyId("fr-FR-Standard".into())).expect("French");
        let output = phonemicizer
            .phonemicize(&request("vous voulez", "fr-FR-Standard"))
            .expect("French should phonemicize");

        assert_eq!(phoneme_symbols(&output), ["v", "u", "v", "u", "l", "e"]);
        assert_eq!(
            output.phonemes[0].provenance.source,
            EvidenceSource::Lexicon
        );
    }

    #[test]
    fn french_pronounces_monsieur_as_contracted_title() {
        let phonemicizer =
            phonemicizer_for_variety(&VarietyId("fr-FR-Standard".into())).expect("French");
        let monsieur = phonemicizer
            .phonemicize(&request("Monsieur", "fr-FR-Standard"))
            .expect("French title should phonemicize");
        let monsieur_symbols = phoneme_symbols(&monsieur);

        assert_eq!(monsieur_symbols, ["m", "ə", "s", "j", "ø"]);
        assert_eq!(
            monsieur.phonemes[0].provenance.source,
            EvidenceSource::Lexicon
        );
        assert!(!monsieur_symbols.contains(&"ɔ̃".to_string()));
        assert!(!monsieur_symbols.contains(&"ʁ".to_string()));

        let monseigneur = phonemicizer
            .phonemicize(&request("monseigneur", "fr-FR-Standard"))
            .expect("French title should phonemicize");
        let monseigneur_symbols = phoneme_symbols(&monseigneur);

        assert!(monseigneur_symbols.contains(&"ɔ̃".to_string()));
        assert!(monseigneur_symbols.contains(&"ɲ".to_string()));
    }

    #[test]
    fn french_pronounces_myriel_with_yod() {
        let phonemicizer =
            phonemicizer_for_variety(&VarietyId("fr-FR-Standard".into())).expect("French");
        let output = phonemicizer
            .phonemicize(&request("Monsieur Myriel", "fr-FR-Standard"))
            .expect("French proper name should phonemicize");

        assert_eq!(
            phoneme_symbols(&output),
            ["m", "ə", "s", "j", "ø", "m", "i", "ʁ", "j", "ɛ", "l"]
        );
        assert!(
            output
                .phonemes
                .iter()
                .all(|token| token.provenance.source == EvidenceSource::Lexicon)
        );
    }

    #[test]
    fn french_rule_fallback_handles_regular_final_ez() {
        let phonemicizer =
            phonemicizer_for_variety(&VarietyId("fr-FR-Standard".into())).expect("French");
        let output = phonemicizer
            .phonemicize(&request("parlez", "fr-FR-Standard"))
            .expect("French fallback should phonemicize");

        assert_eq!(phoneme_symbols(&output), ["p", "a", "ʁ", "l", "e"]);
    }

    #[test]
    fn french_syntax_marks_ent_verbs_for_silent_ending() {
        let phonemicizer =
            phonemicizer_for_variety(&VarietyId("fr-FR-Standard".into())).expect("French");
        let output = phonemicizer
            .phonemicize(&request("ils parlent", "fr-FR-Standard"))
            .expect("French should phonemicize with syntax context");

        assert_eq!(output.syntax.tokens[1].pos, PartOfSpeech::Verb);
        assert!(
            !phoneme_symbols(&output).contains(&"ɑ̃".to_string()),
            "{:?}",
            phoneme_symbols(&output)
        );
    }

    #[test]
    fn french_famous_line_keeps_pense_final_s() {
        let phonemicizer =
            phonemicizer_for_variety(&VarietyId("fr-FR-Standard".into())).expect("French");
        let output = phonemicizer
            .phonemicize(&request("Je pense, donc je suis.", "fr-FR-Standard"))
            .expect("French famous line should phonemicize");

        assert_eq!(
            phoneme_symbols(&output),
            ["ʒ", "p", "ɑ̃", "s", "d", "ɔ̃", "k", "ʒ", "s", "ɥ", "i"]
        );
    }

    #[test]
    fn french_connected_speech_adds_liaison_and_deletes_final_schwa() {
        let phonemicizer =
            phonemicizer_for_variety(&VarietyId("fr-FR-Standard".into())).expect("French");
        let liaison = phonemicizer
            .phonemicize(&request("vous avez", "fr-FR-Standard"))
            .expect("French liaison should phonemicize");
        assert!(
            phone_symbols(&liaison)
                .windows(4)
                .any(|window| window == ["v", "u", "z", "|"]),
            "{:?}",
            phone_symbols(&liaison)
        );

        let schwa = phonemicizer
            .phonemicize(&request("le garçon", "fr-FR-Standard"))
            .expect("French schwa deletion should phonemicize");
        assert!(
            phone_symbols(&schwa)
                .windows(2)
                .any(|window| window == ["l", "|"]),
            "{:?}",
            phone_symbols(&schwa)
        );
    }

    #[test]
    fn builtin_non_english_number_normalizers_spell_small_numbers() {
        assert_eq!(
            normalize_small_numbers_for_variety(
                "J'ai 3 amis.",
                &VarietyId("fr-FR-Standard".into())
            ),
            "J'ai trois amis."
        );
        assert_eq!(
            normalize_small_numbers_for_variety(
                "Tengo 12 gatos.",
                &VarietyId("es-419-Standard".into())
            ),
            "Tengo doce gatos."
        );
        assert_eq!(
            normalize_small_numbers_for_variety(
                "Ich habe 5 Katzen.",
                &VarietyId("de-DE-Standard".into())
            ),
            "Ich habe fünf Katzen."
        );
        assert_eq!(
            normalize_small_numbers_for_variety("Mi havas 2 katojn.", &VarietyId("eo".into())),
            "Mi havas du katojn."
        );
        assert_eq!(
            normalize_small_numbers_for_variety(
                "Habeo 2 feles.",
                &VarietyId("la-Classical".into())
            ),
            "Habeo duo feles."
        );
        assert_eq!(
            normalize_small_numbers_for_variety(
                "Έχω 2 γάτες.",
                &VarietyId("el-GR-Standard".into())
            ),
            "Έχω δύο γάτες."
        );
        assert_eq!(
            normalize_small_numbers_for_variety("2 granthau", &VarietyId("san".into())),
            "dvi granthau"
        );
        assert_eq!(
            normalize_small_numbers_for_variety("A2 restas kodo.", &VarietyId("eo".into())),
            "A2 restas kodo."
        );
    }

    #[test]
    fn roman_numerals_normalize_with_variety_number_names() {
        assert_eq!(
            normalize_text_for_variety(
                "Chapitre I Monsieur Myriel. Chapitre XII Solitude.",
                &VarietyId("fr-FR-Standard".into())
            ),
            "Chapitre un Monsieur Myriel. Chapitre douze Solitude."
        );
        assert_eq!(
            normalize_text_for_variety("Louis XIV arrive.", &VarietyId("fr-FR-Standard".into())),
            "Louis quatorze arrive."
        );
        assert_eq!(
            normalize_text_for_variety("I read Chapter I.", &VarietyId("en-US".into())),
            "I read Chapter one."
        );
        assert_eq!(
            normalize_text_for_variety("I saw V shapes.", &VarietyId("en-US".into())),
            "I saw V shapes."
        );
        assert_eq!(
            normalize_text_for_variety("The CD player works.", &VarietyId("en-US".into())),
            "The CD player works."
        );
        assert_eq!(
            normalize_text_for_variety("Volume CD closes.", &VarietyId("en-US".into())),
            "Volume four hundred closes."
        );
    }

    #[test]
    fn french_general_number_normalization_handles_years_and_dates() {
        assert_eq!(
            normalize_text_for_variety(
                "En 1815, il avait 75 ans depuis 1806.",
                &VarietyId("fr-FR-Standard".into())
            ),
            "En un mille huit cent quinze, il avait soixante-dix-cinq ans depuis un mille huit cent six."
        );
        assert_eq!(
            normalize_text_for_variety("2024-06-20", &VarietyId("fr-FR-Standard".into())),
            "vingt vingt-quatre tiret zéro six tiret vingt"
        );
    }

    #[test]
    fn french_initialisms_use_french_letter_names() {
        let output = phonemicizer_for_variety(&VarietyId("fr-FR-Standard".into()))
            .expect("French phonemicizer")
            .phonemicize(&request("A et B", "fr-FR-Standard"))
            .expect("French letters should phonemicize");
        assert_eq!(phoneme_symbols(&output), ["a", "e", "b", "e"]);
        assert!(
            output.phonemes.iter().any(|token| {
                phoneme_usize_feature(token, "orthography.letter_index").is_some()
            }),
            "{:?}",
            output.phonemes
        );
    }

    #[test]
    fn digit_expansion_phonemicizes_for_every_builtin_variety() {
        for variety in builtin_varieties() {
            let output = phonemicizer_for_variety(&variety.id)
                .expect("phonemicizer")
                .phonemicize(&request("2", &variety.id.0))
                .unwrap_or_else(|err| panic!("{} should phonemicize digit: {err}", variety.id.0));
            assert!(
                !output.phonemes.is_empty(),
                "{} should produce phonemes for digit expansion",
                variety.id.0
            );
            assert!(
                output.warnings.iter().all(|warning| {
                    warning.kind != PronunciationWarningKind::UnknownPronunciation
                }),
                "{} should not report unknown digit pronunciation: {:?}",
                variety.id.0,
                output.warnings
            );
        }
    }

    #[test]
    fn orthographic_pronunciation_phonemicizes_every_builtin_variety() {
        for variety in builtin_varieties() {
            let sample = sample_word_for_variety(&variety);
            let output = phonemicizer_for_variety(&variety.id)
                .expect("phonemicizer")
                .phonemicize(&request(sample, &variety.id.0))
                .unwrap_or_else(|err| {
                    panic!(
                        "{} should phonemicize sample `{sample}`: {err}",
                        variety.id.0
                    )
                });
            assert!(
                !output.phonemes.is_empty(),
                "{} should produce phonemes for sample `{sample}`",
                variety.id.0
            );
            assert!(
                output
                    .warnings
                    .iter()
                    .all(|warning| warning.kind != PronunciationWarningKind::UnknownPronunciation),
                "{} should phonemicize sample `{sample}` from its declared data: {:?}",
                variety.id.0,
                output.warnings
            );
        }
    }

    #[test]
    fn generic_pipeline_does_not_use_english_unknown_guessing() {
        let english = VarietyDataPhonemicizer
            .phonemicize(&request("zzq", "en-US"))
            .expect("English variety-data pipeline should report unknown words");
        assert!(
            english
                .warnings
                .iter()
                .any(|warning| warning.kind == PronunciationWarningKind::UnknownPronunciation),
            "{:?}",
            english.warnings
        );
        assert!(
            english
                .warnings
                .iter()
                .all(|warning| warning.kind != PronunciationWarningKind::GuessedWord),
            "{:?}",
            english.warnings
        );
        assert!(
            english.phonemes.is_empty(),
            "generic pipeline should not synthesize English-only fallback phonemes"
        );
    }

    #[test]
    fn punctuation_and_question_contours_are_variety_data() {
        for variety in builtin_varieties() {
            assert!(
                variety.punctuation.is_some(),
                "{} should carry punctuation boundary data",
                variety.id.0
            );
            assert!(
                variety.question_contours.is_some(),
                "{} should carry question contour data",
                variety.id.0
            );
        }

        let english = variety_by_code("en-US").expect("English variety");
        let english_words = tokenize_words("Dr. Smith left.");
        let english_boundaries = boundary_tokens("Dr. Smith left.", &english_words, &english);
        assert!(
            english_boundaries
                .iter()
                .all(|boundary| boundary.after_grapheme_index != 0 || boundary.terminal.is_none()),
            "English title abbreviation should suppress a terminal boundary: {english_boundaries:?}"
        );

        let french = variety_by_code("fr").expect("French variety");
        let french_words = tokenize_words("Dr. Dupont part.");
        let french_boundaries = boundary_tokens("Dr. Dupont part.", &french_words, &french);
        assert!(
            french_boundaries
                .iter()
                .all(|boundary| boundary.after_grapheme_index != 0 || boundary.terminal.is_none()),
            "French should use declared abbreviation data: {french_boundaries:?}"
        );

        let sanskrit = variety_by_code("san").expect("Sanskrit variety");
        let sanskrit_words = tokenize_words("Dr. Smith left.");
        let sanskrit_boundaries = boundary_tokens("Dr. Smith left.", &sanskrit_words, &sanskrit);
        assert!(
            sanskrit_boundaries.iter().any(|boundary| {
                boundary.after_grapheme_index == 0
                    && boundary.terminal == Some(TerminalPunctuation::Period)
            }),
            "English abbreviation data should not leak into Sanskrit: {sanskrit_boundaries:?}"
        );

        let english_questions = english
            .question_contours
            .as_ref()
            .expect("English question contours");
        assert!(english_questions.wh_openers.contains(&"what".into()));
        assert!(!english_questions.wh_openers.contains(&"qué".into()));

        let spanish = variety_by_code("es").expect("Spanish variety");
        let spanish_questions = spanish
            .question_contours
            .as_ref()
            .expect("Spanish question contours");
        assert!(spanish_questions.wh_openers.contains(&"qué".into()));
        assert!(!spanish_questions.wh_openers.contains(&"what".into()));
    }

    #[test]
    fn non_english_question_contours_are_used_from_variety_data() {
        let spanish = phonemicizer_for_variety(&VarietyId("es".into()))
            .expect("Spanish phonemicizer")
            .phonemicize(&request("qué casa?", "es"))
            .expect("Spanish wh question should phonemicize");
        assert!(
            spanish
                .prosody
                .labels
                .iter()
                .any(|label| label.kind == ProsodicLabelKind::FinalFall),
            "Spanish wh question should use Spanish contour data: {:?}",
            spanish.prosody.labels
        );

        let esperanto = phonemicizer_for_variety(&VarietyId("eo".into()))
            .expect("Esperanto phonemicizer")
            .phonemicize(&request("ĉu vi?", "eo"))
            .expect("Esperanto yes/no question should phonemicize");
        assert!(
            esperanto
                .prosody
                .labels
                .iter()
                .any(|label| label.kind == ProsodicLabelKind::QuestionRise),
            "Esperanto yes/no question should use Esperanto contour data: {:?}",
            esperanto.prosody.labels
        );
    }

    #[test]
    fn builtin_varieties_expose_runtime_data_hooks() {
        for variety in builtin_varieties() {
            assert!(
                !matches!(
                    variety.text_normalization.number_normalization,
                    NumberNormalizationProfile::None
                ),
                "{} should declare text normalization data",
                variety.id.0
            );
            if matches!(
                variety.text_normalization.number_normalization,
                NumberNormalizationProfile::General
            ) {
                let names = variety
                    .number_names
                    .as_ref()
                    .unwrap_or_else(|| panic!("{} should declare number names", variety.id.0));
                assert!(
                    names.decimal_separator_name.is_some()
                        && names.clock_zero_minute_name.is_some()
                        && names.clock_leading_zero_name.is_some()
                        && names.range_separator_name.is_some()
                        && names.product_separator_name.is_some()
                        && names.slash_separator_name.is_some()
                        && names.date_separator_name.is_some()
                        && names.currency_major_singular.is_some()
                        && names.currency_major_plural.is_some()
                        && names.currency_minor_singular.is_some()
                        && names.currency_minor_plural.is_some()
                        && !names.ordinal_suffixes.is_empty(),
                    "{} should declare every general number-normalization label it uses",
                    variety.id.0
                );
            }
            assert!(
                variety.syntax_analyzer.is_some() || variety.syntax_rules.is_some(),
                "{} should carry syntax analysis data",
                variety.id.0
            );
            if variety.language.0 != "en" {
                assert!(
                    variety
                        .orthography_pronunciation
                        .and_then(|rules| rules.synthesize_ipa)
                        .is_some(),
                    "{} should carry an orthography IPA synthesizer",
                    variety.id.0
                );
            }

            for lexicon_id in &variety.pronunciation_lexicons {
                assert!(
                    lexicons::adapter_for_id(lexicon_id).is_some(),
                    "{} declares unknown pronunciation lexicon `{}`",
                    variety.id.0,
                    lexicon_id
                );
            }

            for rule in &variety.weak_forms {
                assert!(
                    rule.source_pronunciation.is_empty()
                        || source_pronunciation_notation(
                            rule.source_pronunciation_notation.as_deref()
                        )
                        .is_some(),
                    "{} weak form `{}` has source pronunciation without declared notation",
                    variety.id.0,
                    rule.id
                );
            }

            for entry in &variety.orthographic_unit_pronunciations {
                assert!(
                    entry.source_pronunciation.is_empty()
                        || source_pronunciation_notation(
                            entry.source_pronunciation_notation.as_deref()
                        )
                        .is_some(),
                    "{} orthographic unit `{}` has source pronunciation without declared notation",
                    variety.id.0,
                    entry.unit
                );
            }

            for rule in &variety.pronunciation_selection_rules {
                assert!(
                    rule.source_pronunciation.is_empty()
                        || source_pronunciation_notation(
                            rule.source_pronunciation_notation.as_deref()
                        )
                        .is_some(),
                    "{} pronunciation rule `{}` has source pronunciation without declared notation",
                    variety.id.0,
                    rule.lexical_item
                );
            }

            if let Some(pipeline_id) = variety.pronunciation_pipeline.as_deref() {
                assert!(
                    PRONUNCIATION_PIPELINE_REGISTRY
                        .iter()
                        .any(|registration| registration.id == pipeline_id),
                    "{} declares unknown pronunciation pipeline `{}`",
                    variety.id.0,
                    pipeline_id
                );
            }
        }
    }

    #[test]
    fn generic_pipeline_uses_variety_declared_text_normalization() {
        let english =
            phonemicizer_for_variety(&VarietyId("en-US".into())).expect("English phonemicizer");
        let output = english
            .phonemicize(&request("Dr. Smith saw No. 5 at 3:05 p.m.", "en-US"))
            .expect("English variety-data normalizer should phonemicize");
        assert_eq!(
            output
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            [
                "Doctor", "Smith", "saw", "Number", "five", "at", "three", "oh", "five", "p", "m"
            ]
        );

        let french =
            phonemicizer_for_variety(&VarietyId("fr".into())).expect("French phonemicizer");
        let output = french
            .phonemicize(&request("J'ai 5 amis.", "fr"))
            .expect("French small-number normalizer should phonemicize");
        assert_eq!(
            output
                .graphemes
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            ["J'ai", "cinq", "amis"]
        );
    }

    fn sample_word_for_variety(variety: &LinguisticVariety) -> &str {
        variety
            .orthography
            .as_ref()
            .and_then(|orthography| orthography.sample_words.first())
            .map(String::as_str)
            .unwrap_or_else(|| panic!("{} should declare orthography sample words", variety.id.0))
    }

    fn sample_letter_for_variety(variety: &LinguisticVariety, index: usize) -> &str {
        variety
            .orthography
            .as_ref()
            .and_then(|orthography| orthography.sample_letter_units.get(index))
            .map(String::as_str)
            .unwrap_or_else(|| {
                panic!(
                    "{} should declare at least {} orthography sample letters",
                    variety.id.0,
                    index + 1
                )
            })
    }

    #[test]
    fn every_builtin_variety_phonemicizes_with_declared_data() {
        for variety in builtin_varieties() {
            let word = sample_word_for_variety(&variety);
            let phonemicizer =
                phonemicizer_for_variety(&variety.id).expect("builtin variety has phonemicizer");
            let output = phonemicizer
                .phonemicize(&request(word, &variety.id.0))
                .unwrap_or_else(|error| panic!("{} should phonemicize: {error}", variety.id.0));
            assert_eq!(output.variety, variety.id);
            assert!(
                !output.phonemes.is_empty(),
                "{} should produce phonemes for {word}",
                variety.id.0
            );
            assert!(
                output
                    .phonemes
                    .iter()
                    .any(|token| !matches!(token.phoneme, Spec::Known(ref id) if id.0.starts_with("boundary."))),
                "{} should produce non-boundary phonemes for {word}",
                variety.id.0
            );
        }
    }

    #[test]
    fn initialism_joiners_and_mixed_tokens_are_variety_data_for_every_builtin_variety() {
        for variety in builtin_varieties() {
            let orthography = variety
                .orthography
                .as_ref()
                .expect("builtin variety should declare orthography data");
            let joiner = orthography
                .initialism_joiners
                .first()
                .unwrap_or_else(|| panic!("{} should declare initialism joiners", variety.id.0));
            let left = sample_letter_for_variety(&variety, 0);
            let right = sample_letter_for_variety(&variety, 1);
            let initialism_text = format!("{left} {joiner} {right}");
            let initialism = phonemicizer_for_variety(&variety.id)
                .expect("phonemicizer")
                .phonemicize(&request(&initialism_text, &variety.id.0))
                .unwrap_or_else(|error| {
                    panic!(
                        "{} should phonemicize initialism `{initialism_text}`: {error}",
                        variety.id.0
                    )
                });
            assert!(
                initialism.phonemes.iter().any(|token| {
                    phoneme_usize_feature(token, "orthography.letter_index").is_some()
                }),
                "{} should mark letter-name phonemes for `{initialism_text}`: {:?}",
                variety.id.0,
                phoneme_symbols(&initialism)
            );

            let mixed_text = format!("{left}2");
            let mixed = phonemicizer_for_variety(&variety.id)
                .expect("phonemicizer")
                .phonemicize(&request(&mixed_text, &variety.id.0))
                .unwrap_or_else(|error| {
                    panic!(
                        "{} should phonemicize mixed token `{mixed_text}`: {error}",
                        variety.id.0
                    )
                });
            assert!(
                mixed
                    .warnings
                    .iter()
                    .all(|warning| warning.kind != PronunciationWarningKind::UnknownPronunciation),
                "{} should expand mixed token `{mixed_text}` from variety data: {:?}",
                variety.id.0,
                mixed.warnings
            );
            assert!(
                mixed.phonemes.iter().any(|token| {
                    phoneme_usize_feature(token, "orthography.letter_index").is_some()
                }),
                "{} should mark letter-name phonemes for mixed token `{mixed_text}`",
                variety.id.0
            );
        }
    }

    #[test]
    fn builtin_non_english_varieties_phonemicize_from_variety_data() {
        let spanish = phonemicizer_for_variety(&VarietyId("es".into()))
            .expect("Spanish phonemicizer")
            .phonemicize(&request("pato", "es"))
            .expect("Spanish should phonemicize from variety aliases");
        assert_eq!(spanish.variety.0, "es");
        assert_eq!(phoneme_symbols(&spanish), vec!["p", "a", "t", "o"]);
        assert_eq!(spanish.syntax.link_parses.len(), 1);

        let castilian = phonemicizer_for_variety(&VarietyId("es-ES-Castilian".into()))
            .expect("Castilian Spanish phonemicizer")
            .phonemicize(&request("zapato", "es-ES-Castilian"))
            .expect("Castilian Spanish should phonemicize from spelling rules");
        let latam = phonemicizer_for_variety(&VarietyId("es-419-Standard".into()))
            .expect("Latin American Spanish phonemicizer")
            .phonemicize(&request("zapato", "es-419-Standard"))
            .expect("Latin American Spanish should phonemicize from spelling rules");
        assert_eq!(
            phoneme_symbols(&castilian),
            vec!["θ", "a", "p", "a", "t", "o"]
        );
        assert_eq!(phoneme_symbols(&latam), vec!["s", "a", "p", "a", "t", "o"]);

        let esperanto = phonemicizer_for_variety(&VarietyId("eo".into()))
            .expect("Esperanto phonemicizer")
            .phonemicize(&request("ŝipo", "eo"))
            .expect("Esperanto should phonemicize from spelling rules");
        assert_eq!(phoneme_symbols(&esperanto), vec!["ʃ", "i", "p", "o"]);

        let french = phonemicizer_for_variety(&VarietyId("fra".into()))
            .expect("French phonemicizer")
            .phonemicize(&request("bonjour", "fra"))
            .expect("French should phonemicize from spelling rules");
        assert_eq!(phoneme_symbols(&french), vec!["b", "ɔ̃", "ʒ", "u", "ʁ"]);
        assert_eq!(french.syntax.link_parses.len(), 1);

        let german = phonemicizer_for_variety(&VarietyId("deu".into()))
            .expect("German phonemicizer")
            .phonemicize(&request("Sprache", "deu"))
            .expect("German should phonemicize from spelling rules");
        assert_eq!(phoneme_symbols(&german), vec!["ʃ", "p", "r", "a", "x", "ə"]);

        let classical = phonemicizer_for_variety(&VarietyId("la-Classical".into()))
            .expect("Classical Latin phonemicizer")
            .phonemicize(&request("caelum", "la-Classical"))
            .expect("Classical Latin should phonemicize from spelling rules");
        let ecclesiastical = phonemicizer_for_variety(&VarietyId("la-Ecclesiastical".into()))
            .expect("Ecclesiastical Latin phonemicizer")
            .phonemicize(&request("caelum", "la-Ecclesiastical"))
            .expect("Ecclesiastical Latin should phonemicize from spelling rules");
        assert_eq!(phoneme_symbols(&classical), vec!["k", "ae̯", "l", "u", "m"]);
        assert_eq!(
            phoneme_symbols(&ecclesiastical),
            vec!["t͡ʃ", "ae", "l", "u", "m"]
        );

        let modern_greek = phonemicizer_for_variety(&VarietyId("el-GR-Standard".into()))
            .expect("Modern Greek phonemicizer")
            .phonemicize(&request("και", "el-GR-Standard"))
            .expect("Modern Greek should phonemicize from spelling rules");
        let ancient_greek = phonemicizer_for_variety(&VarietyId("grc-Attic".into()))
            .expect("Ancient Greek phonemicizer")
            .phonemicize(&request("και", "grc-Attic"))
            .expect("Ancient Greek should phonemicize from spelling rules");
        let koine_greek = phonemicizer_for_variety(&VarietyId("grc-Koine".into()))
            .expect("Koine Greek phonemicizer")
            .phonemicize(&request("και", "grc-Koine"))
            .expect("Koine Greek should phonemicize from spelling rules");
        assert_eq!(phoneme_symbols(&modern_greek), vec!["c", "e"]);
        assert_eq!(phoneme_symbols(&ancient_greek), vec!["k", "ai̯"]);
        assert_eq!(phoneme_symbols(&koine_greek), vec!["k", "e"]);

        let sanskrit = phonemicizer_for_variety(&VarietyId("san".into()))
            .expect("Sanskrit phonemicizer")
            .phonemicize(&request("धर्म", "san"))
            .expect("Sanskrit should phonemicize from spelling rules");
        assert_eq!(phoneme_symbols(&sanskrit), vec!["dʱ", "a", "r", "m", "a"]);
    }

    #[test]
    fn unknown_word_missing_is_explicitly_marked() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("zzq", "en-US"))
            .expect("unknown word should be reported");

        assert!(output.phonemes.is_empty());
        assert!(output.warnings.iter().any(|warning| {
            warning.kind == PronunciationWarningKind::UnknownPronunciation
                && warning.message.contains("unknown pronunciation")
        }));
    }

    #[test]
    fn punctuation_emits_typed_boundaries() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("hello, world?", "en-US"))
            .expect("punctuated text should phonemicize");

        assert!(output.boundaries.iter().any(|boundary| {
            boundary.kind == BoundaryKind::Phrase
                && boundary.after_grapheme_index == 0
                && boundary.pause == Some(PauseKind::Comma)
        }));
        assert!(output.boundaries.iter().any(|boundary| {
            boundary.kind == BoundaryKind::Phrase
                && boundary.after_grapheme_index == 1
                && boundary.terminal == Some(TerminalPunctuation::Question)
        }));
        assert!(output.prosody.labels.iter().any(|label| {
            label.kind == ProsodicLabelKind::ContinuationRise && label.confidence > 0.0
        }));
        assert!(output.prosody.labels.iter().any(|label| {
            label.kind == ProsodicLabelKind::QuestionRise && label.confidence > 0.0
        }));
    }

    #[test]
    fn test_abbreviation_periods_are_not_terminal_boundaries() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("About 17,000 cases will stay at 630 Sansome St. in San Francisco, another, smaller location with just two operating courtrooms.", "en-US"))
            .expect("should phonemicize");

        let st_boundary = output.boundaries.iter().find(|b| {
            output
                .graphemes
                .get(b.after_grapheme_index)
                .is_some_and(|g| g.text == "St")
        });
        assert!(
            st_boundary.is_some_and(|b| b.terminal.is_none()),
            "St. followed by lowercase should not be a sentence boundary"
        );

        let output2 = VarietyDataPhonemicizer
            .phonemicize(&request(
                "He lives on Sansome St. The house is blue.",
                "en-US",
            ))
            .expect("should phonemicize");
        let st_boundary2 = output2.boundaries.iter().find(|b| {
            output2
                .graphemes
                .get(b.after_grapheme_index)
                .is_some_and(|g| g.text == "St")
        });
        assert!(
            st_boundary2.is_some_and(|b| b.terminal == Some(TerminalPunctuation::Period)),
            "St. followed by sentence starter should be a sentence boundary"
        );

        let output3 = VarietyDataPhonemicizer
            .phonemicize(&request("We visited St. Charles.", "en-US"))
            .expect("should phonemicize");
        let st_boundary3 = output3.boundaries.iter().find(|b| {
            output3
                .graphemes
                .get(b.after_grapheme_index)
                .is_some_and(|g| g.text == "St")
        });
        assert!(
            st_boundary3.is_none() || st_boundary3.is_some_and(|b| b.terminal.is_none()),
            "St. Charles should not have a terminal period after St."
        );

        let output4 = VarietyDataPhonemicizer
            .phonemicize(&request("He lives on Sansome St.", "en-US"))
            .expect("should phonemicize");
        let st_boundary4 = output4.boundaries.iter().find(|b| {
            output4
                .graphemes
                .get(b.after_grapheme_index)
                .is_some_and(|g| g.text == "St")
        });
        assert!(
            st_boundary4.is_some_and(|b| b.terminal == Some(TerminalPunctuation::Period)),
            "St. at the end of the text should be a sentence boundary"
        );
    }

    #[test]
    fn loadstone_is_pronounced_like_lodestone() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("The Loadstone Rock was drawing him.", "en-US"))
            .expect("should phonemicize");
        let symbols = phoneme_symbols(&output).join("");
        assert!(symbols.contains("loʊdstoʊn"), "{symbols}");
        assert!(!symbols.contains("lʌədstəni"), "{symbols}");
        assert!(!symbols.contains("ləədstəni"), "{symbols}");
    }

    #[test]
    fn st_name_prefix_is_saint_but_street_abbreviation_stays_street() {
        let saint_output = VarietyDataPhonemicizer
            .phonemicize(&request("We visited St. Charles.", "en-US"))
            .expect("should phonemicize");
        let saint_symbols = cmudict_symbols(&saint_output).join(" ");
        assert!(saint_symbols.contains("S EY1 N T"), "{saint_symbols}");
        assert!(!saint_symbols.contains("S T R IY1 T"), "{saint_symbols}");

        let street_output = VarietyDataPhonemicizer
            .phonemicize(&request(
                "He lives on Sansome St. The house is blue.",
                "en-US",
            ))
            .expect("should phonemicize");
        let street_symbols = cmudict_symbols(&street_output).join(" ");
        assert!(street_symbols.contains("S T R IY1 T"), "{street_symbols}");
    }

    #[test]
    fn yes_no_questions_get_rising_prosody() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("Are you coming?", "en-US"))
            .expect("yes/no question should phonemicize");

        assert!(output.prosody.labels.iter().any(|label| {
            label.kind == ProsodicLabelKind::QuestionRise && label.confidence > 0.0
        }));
        assert!(
            !output
                .prosody
                .labels
                .iter()
                .any(|label| label.kind == ProsodicLabelKind::AlternativeQuestionFall)
        );
    }

    #[test]
    fn wh_questions_do_not_get_yes_no_question_rise() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("What did you choose?", "en-US"))
            .expect("wh question should phonemicize");

        assert!(
            output.prosody.labels.iter().any(|label| {
                label.kind == ProsodicLabelKind::FinalFall && label.confidence > 0.0
            })
        );
        assert!(
            !output
                .prosody
                .labels
                .iter()
                .any(|label| label.kind == ProsodicLabelKind::QuestionRise)
        );
    }

    #[test]
    fn either_or_questions_get_alternative_question_fall() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("Do you want either tea or coffee?", "en-US"))
            .expect("alternative question should phonemicize");

        assert!(output.boundaries.iter().any(|boundary| {
            boundary.kind == BoundaryKind::Phrase
                && boundary.terminal == Some(TerminalPunctuation::Question)
        }));
        assert!(output.prosody.labels.iter().any(|label| {
            label.kind == ProsodicLabelKind::AlternativeQuestionFall && label.confidence > 0.0
        }));
        assert!(
            !output
                .prosody
                .labels
                .iter()
                .any(|label| label.kind == ProsodicLabelKind::QuestionRise)
        );
    }

    #[test]
    fn would_you_rather_questions_rise_on_first_linked_option_and_fall_at_end() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request(
                "Would you rather marry or fly an airplane?",
                "en-US",
            ))
            .expect("alternative question should phonemicize");

        assert!(
            output
                .syntax
                .word_has_link(3, SyntacticLinkKind::Coordination)
        );
        assert!(
            output
                .syntax
                .word_has_link(5, SyntacticLinkKind::Coordination)
        );
        assert!(output.boundaries.iter().any(|boundary| {
            boundary.kind == BoundaryKind::Phrase
                && boundary.after_grapheme_index == 3
                && boundary.pause == Some(PauseKind::AlternativeQuestionRise)
        }));
        assert!(output.prosody.labels.iter().any(|label| {
            label.kind == ProsodicLabelKind::AlternativeQuestionRise && label.confidence > 0.0
        }));
        assert!(output.prosody.labels.iter().any(|label| {
            label.kind == ProsodicLabelKind::AlternativeQuestionFall && label.confidence > 0.0
        }));
        assert!(
            !output
                .prosody
                .labels
                .iter()
                .any(|label| label.kind == ProsodicLabelKind::QuestionRise)
        );
    }

    #[test]
    fn phonemicize_output_exposes_grammar_parse_for_rule_matching() {
        let output = VarietyDataPhonemicizer
            .phonemicize(&request("Do you want either tea or coffee?", "en-US"))
            .expect("sentence should phonemicize");
        let rule_context = output.syntax.rule_context();

        assert!(output.syntax.word_has_link(0, SyntacticLinkKind::Auxiliary));
        assert!(
            output
                .syntax
                .word_has_link(5, SyntacticLinkKind::Coordination)
        );
        assert!(
            RuleCondition::CurrentWordHasSyntacticLink(SyntacticLinkKind::Auxiliary)
                .matches_syntax(&rule_context, 0)
        );
        assert!(
            RuleCondition::PreviousWordHasSyntacticLink(SyntacticLinkKind::Coordination)
                .matches_syntax(&rule_context, 6)
        );
    }
}
