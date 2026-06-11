use serde::{Deserialize, Serialize};

use crate::acoustics::AcousticProfile;
use crate::data::lexicons::cmudict::CmuPhoneme;
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeakFormRule {
    pub id: String,
    pub lexical_item: String,
    pub pronunciation: Vec<PhonemeId>,
    #[serde(default)]
    pub cmudict_pronunciation: Vec<CmuPhoneme>,
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
    pub cmudict_pronunciation: Vec<CmuPhoneme>,
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
