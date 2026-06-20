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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Language {
    pub id: LanguageId,
    pub name: String,
    pub endonym: Option<String>,
    pub iso_639: Option<String>,
}

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
    pub pronunciation_pipeline: Option<String>,
    #[serde(default)]
    pub syntax_profile: Option<String>,
    #[serde(default)]
    pub number_names: Option<NumberNameSet>,
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
    pub ordinal_suffixes: Vec<OrdinalSuffixName>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrdinalSuffixName {
    pub value: u32,
    pub suffixes: Vec<String>,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ConnectedSpeechRule {
    DeleteFinalPhoneBeforeConsonant {
        phone: String,
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
