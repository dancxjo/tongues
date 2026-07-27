use serde::{Deserialize, Serialize};

use crate::acoustics::AcousticProfile;
use crate::feature::FeatureSystem;
use crate::ids::{LanguageId, PhonemeId, VarietyId};
use crate::morphology::Morphology;
use crate::orthography::Orthography;
use crate::phonetics::PhoneInventory;
use crate::phonology::PhonemeInventory;
use crate::prosody::ProsodyProfile;
use crate::rules::{AllophoneRule, EpenthesisRule, Phonotactics};
use crate::segment::TerminalPunctuation;
use crate::syntax::{GrammarRuleSet, PartOfSpeech, SentenceSyntaxAnalysis};

pub type SyntaxAnalyzer = fn(&[String], Option<TerminalPunctuation>) -> SentenceSyntaxAnalysis;
pub type OrthographyIpaSynthesizer =
    fn(&str, &LinguisticVariety, Option<PartOfSpeech>) -> Option<String>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Language {
    pub id: LanguageId,
    pub name: String,
    pub endonym: Option<String>,
    pub iso_639: Option<String>,
}

// Equality is used for these data descriptors, including their static callback
// identity. Preserve that public behavior while keeping the allowance local.
#[allow(unpredictable_function_pointer_comparisons)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinguisticVariety {
    pub id: VarietyId,
    pub language: LanguageId,
    pub name: String,
    pub feature_system: FeatureSystem,
    pub phonemes: PhonemeInventory,
    pub phones: PhoneInventory,
    pub allophone_rules: Vec<AllophoneRule>,
    #[serde(default)]
    pub epenthesis_rules: Vec<EpenthesisRule>,
    #[serde(default)]
    pub weak_forms: Vec<WeakFormRule>,
    #[serde(default)]
    pub orthographic_unit_pronunciations: Vec<OrthographicUnitPronunciation>,
    #[serde(default)]
    pub pronunciation_lexicons: Vec<String>,
    #[serde(default)]
    pub pronunciation_selection_rules: Vec<PronunciationSelectionRule>,
    #[serde(default)]
    pub pronunciation_pipeline: Option<String>,
    #[serde(default)]
    pub text_normalization: TextNormalizationProfile,
    #[serde(default)]
    pub syntax_profile: Option<String>,
    #[serde(skip)]
    pub syntax_analyzer: Option<SyntaxAnalyzer>,
    #[serde(skip)]
    pub syntax_rules: Option<GrammarRuleSet>,
    #[serde(skip)]
    pub orthography_pronunciation: Option<OrthographyPronunciationRules>,
    #[serde(default)]
    pub number_names: Option<NumberNameSet>,
    #[serde(default)]
    pub punctuation: Option<PunctuationProfile>,
    #[serde(default)]
    pub question_contours: Option<QuestionContourProfile>,
    #[serde(default)]
    pub connected_speech: Vec<ConnectedSpeechRule>,
    pub phonotactics: Option<Phonotactics>,
    pub orthography: Option<Orthography>,
    pub morphology: Option<Morphology>,
    pub acoustic_profile: Option<AcousticProfile>,
    pub prosody_profile: Option<ProsodyProfile>,
    pub status: VarietyStatus,
    pub implementation_status: VarietyImplementationStatus,
}

// This callback-bearing descriptor intentionally retains its public equality
// implementation; the callback is always selected from static variety data.
#[allow(unpredictable_function_pointer_comparisons)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrthographyPronunciationRules {
    pub synthesize_ipa: Option<OrthographyIpaSynthesizer>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VarietyStatus {
    Attested,
    Reconstructed,
    Pedagogical,
    Experimental,
    Idiolect,
    SessionLocal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type", content = "data")]
pub enum VarietyImplementationStatus {
    Complete,
    StubDerivedFrom(VarietyId),
    PermissiveProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NumberNameSet {
    #[serde(default)]
    pub cardinal_0_to_20: Vec<String>,
    #[serde(default)]
    pub cardinal_tens: Vec<NumberName>,
    #[serde(default)]
    pub hundred_name: Option<String>,
    #[serde(default)]
    pub scale_names: Vec<ScaleName>,
    #[serde(default)]
    pub special_number_names: Vec<NumberName>,
    #[serde(default)]
    pub suffixed_number_names: Vec<SuffixedNumberName>,
    #[serde(default)]
    pub grouped_year_names: Vec<GroupedYearName>,
    #[serde(default)]
    pub year_preceding_words: Vec<String>,
    #[serde(default)]
    pub unit_names: Vec<UnitName>,
    #[serde(default)]
    pub ordinal_suffixes: Vec<OrdinalSuffixName>,
    #[serde(default)]
    pub ordinal_names: Vec<NumberName>,
    #[serde(default)]
    pub decimal_separator_name: Option<String>,
    #[serde(default)]
    pub clock_zero_minute_name: Option<String>,
    #[serde(default)]
    pub clock_leading_zero_name: Option<String>,
    #[serde(default)]
    pub range_separator_name: Option<String>,
    #[serde(default)]
    pub product_separator_name: Option<String>,
    #[serde(default)]
    pub slash_separator_name: Option<String>,
    #[serde(default)]
    pub date_separator_name: Option<String>,
    #[serde(default)]
    pub currency_major_singular: Option<String>,
    #[serde(default)]
    pub currency_major_plural: Option<String>,
    #[serde(default)]
    pub currency_minor_singular: Option<String>,
    #[serde(default)]
    pub currency_minor_plural: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NumberName {
    pub value: u32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuffixedNumberName {
    pub value: u32,
    pub suffixes: Vec<String>,
    pub name: String,
}

/// A language-declared convention for reading a year in two numeric groups.
///
/// For a divisor of 100, `1965` becomes the localized names for `19` and `65`.
/// Exact groups and leading-zero tails use the supplied linking names, so the
/// same normalizer can express conventions such as “nineteen hundred” and
/// “nineteen oh five” without embedding those English words in its logic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupedYearName {
    pub first: u32,
    pub last: u32,
    pub divisor: u32,
    pub exact_group_name: String,
    pub leading_zero_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScaleName {
    pub power: u32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitName {
    pub aliases: Vec<String>,
    pub singular: String,
    pub plural: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrdinalSuffixName {
    pub value: u32,
    pub suffixes: Vec<String>,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TextNormalizationProfile {
    #[serde(default)]
    pub spoken_form_rewrites: Vec<TextRewrite>,
    #[serde(default)]
    pub number_normalization: NumberNormalizationProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextRewrite {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumberNormalizationProfile {
    #[default]
    None,
    SmallNumbers,
    General,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PunctuationProfile {
    #[serde(default)]
    pub period_abbreviations: Vec<String>,
    #[serde(default)]
    pub title_abbreviations: Vec<String>,
    #[serde(default)]
    pub ambiguous_period_abbreviations: Vec<String>,
    #[serde(default)]
    pub sentence_starter_words_after_ambiguous_abbreviation: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct QuestionContourProfile {
    #[serde(default)]
    pub yes_no_openers: Vec<String>,
    #[serde(default)]
    pub wh_openers: Vec<String>,
    #[serde(default)]
    pub alternative_coordinators: Vec<String>,
    #[serde(default)]
    pub paired_alternative_openers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ConnectedSpeechRule {
    DeleteFinalPhoneBeforeConsonant {
        phone: String,
    },
    LinkingR {
        phone: String,
        #[serde(default)]
        intrusive_after_phones: Vec<String>,
    },
    Liaison {
        #[serde(default)]
        entries: Vec<ConnectedSpeechEntry>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectedSpeechEntry {
    pub after_word: String,
    pub before_vowel_phone: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeakFormRule {
    pub id: String,
    pub lexical_item: String,
    pub pronunciation: Vec<PhonemeId>,
    #[serde(default)]
    pub source_pronunciation: Vec<String>,
    #[serde(default)]
    pub source_pronunciation_notation: Option<String>,
    #[serde(default)]
    pub following: WeakFormFollowingContext,
    #[serde(default)]
    pub style: WeakFormStyleContext,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrthographicUnitPronunciation {
    pub kind: OrthographicUnitKind,
    pub unit: String,
    pub pronunciation: Vec<PhonemeId>,
    #[serde(default)]
    pub source_pronunciation: Vec<String>,
    #[serde(default)]
    pub source_pronunciation_notation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationSelectionRule {
    pub lexical_item: String,
    #[serde(default)]
    pub part_of_speech: Option<PartOfSpeech>,
    #[serde(default)]
    pub next_part_of_speech: Option<PartOfSpeech>,
    #[serde(default)]
    pub source_pronunciation: Vec<String>,
    #[serde(default)]
    pub source_pronunciation_notation: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrthographicUnitKind {
    LetterName,
    DigitName,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeakFormFollowingContext {
    #[default]
    Any,
    BeforeVowelish,
    BeforeConsonantish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeakFormStyleContext {
    #[default]
    Any,
    CasualOnly,
}
