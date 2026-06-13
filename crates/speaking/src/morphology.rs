use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, MorphemeId};
use crate::phonology::PhonemeToken;
use crate::spec::Spec;
use crate::time::TextSpan;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MorphemeKind {
    Root,
    Prefix,
    Suffix,
    Infix,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Morpheme {
    pub id: MorphemeId,
    pub form: String,
    pub kind: MorphemeKind,
    pub gloss: Option<String>,
    pub features: FeatureBundle,
    #[serde(default)]
    pub pronunciation: Vec<PhonemeToken>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphemeToken {
    pub morpheme: Spec<MorphemeId>,
    pub surface: String,
    pub span: Option<TextSpan>,
    pub features: FeatureBundle,
    #[serde(default)]
    pub pronunciation: Vec<PhonemeToken>,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Morphology {
    pub morphemes: HashMap<MorphemeId, Morpheme>,
    pub rules: Vec<MorphologicalRule>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphologicalRule {
    pub id: String,
    pub name: String,
    pub triggers: Vec<MorphologicalTrigger>,
    pub actions: Vec<MorphologicalAction>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum MorphologicalTrigger {
    LeftMorphemeId(String),
    RightMorphemeId(String),
    LeftMorphemeKind(MorphemeKind),
    RightMorphemeKind(MorphemeKind),
    LeftEndsWith(String),
    RightStartsWith(String),
    LeftHasFeature { key: String, value: String },
    RightHasFeature { key: String, value: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "data")]
pub enum MorphologicalAction {
    ReplaceLeftSuffix { find: String, replace: String },
    ReplaceRightPrefix { find: String, replace: String },
    DoubleLeftFinalConsonant,
    DropLeftFinalE,
    SetPrimaryStressOnLeftSyllableFromEnd(usize),
    SetPrimaryStressOnRightSyllableFromStart(usize),
    SetSecondaryStressOnLeft,
    SetSecondaryStressOnRight,
    UnstressLeft,
    UnstressRight,
}

impl MorphologicalTrigger {
    pub fn matches(
        &self,
        left: &MorphemeToken,
        right: &MorphemeToken,
        left_meta: &Morpheme,
        right_meta: &Morpheme,
    ) -> bool {
        match self {
            Self::LeftMorphemeId(id) => left_meta.id.0 == *id,
            Self::RightMorphemeId(id) => right_meta.id.0 == *id,
            Self::LeftMorphemeKind(kind) => left_meta.kind == *kind,
            Self::RightMorphemeKind(kind) => right_meta.kind == *kind,
            Self::LeftEndsWith(suffix) => left.surface.ends_with(suffix),
            Self::RightStartsWith(prefix) => right.surface.starts_with(prefix),
            Self::LeftHasFeature { key, value } => {
                let fid = FeatureId(key.clone());
                if let Some(Spec::Known(FeatureValue::Category(val))) =
                    left_meta.features.values.get(&fid)
                {
                    val == value
                } else if let Some(Spec::Known(FeatureValue::Text(val))) =
                    left_meta.features.values.get(&fid)
                {
                    val == value
                } else {
                    false
                }
            }
            Self::RightHasFeature { key, value } => {
                let fid = FeatureId(key.clone());
                if let Some(Spec::Known(FeatureValue::Category(val))) =
                    right_meta.features.values.get(&fid)
                {
                    val == value
                } else if let Some(Spec::Known(FeatureValue::Text(val))) =
                    right_meta.features.values.get(&fid)
                {
                    val == value
                } else {
                    false
                }
            }
        }
    }
}

impl MorphologicalAction {
    pub fn apply(&self, left: &mut MorphemeToken, right: &mut MorphemeToken) {
        match self {
            Self::ReplaceLeftSuffix { find, replace } => {
                if left.surface.ends_with(find) {
                    let new_len = left.surface.len() - find.len();
                    left.surface.truncate(new_len);
                    left.surface.push_str(replace);
                }
            }
            Self::ReplaceRightPrefix { find, replace } => {
                if right.surface.starts_with(find) {
                    right.surface = format!("{}{}", replace, &right.surface[find.len()..]);
                }
            }
            Self::DoubleLeftFinalConsonant => {
                if let Some(last_char) = left.surface.chars().last() {
                    left.surface.push(last_char);
                }
            }
            Self::DropLeftFinalE => {
                if left.surface.ends_with('e') {
                    left.surface.pop();
                }
            }
            Self::SetPrimaryStressOnLeftSyllableFromEnd(n) => {
                let mut vowel_indices = Vec::new();
                for (idx, p) in left.pronunciation.iter().enumerate() {
                    if is_vowel_token(p) {
                        vowel_indices.push(idx);
                    }
                }
                if vowel_indices.len() >= *n {
                    let target = vowel_indices[vowel_indices.len() - *n];
                    set_primary_stress(&mut left.pronunciation, target);
                }
            }
            Self::SetPrimaryStressOnRightSyllableFromStart(n) => {
                let mut vowel_indices = Vec::new();
                for (idx, p) in right.pronunciation.iter().enumerate() {
                    if is_vowel_token(p) {
                        vowel_indices.push(idx);
                    }
                }
                if vowel_indices.len() >= *n {
                    let target = vowel_indices[*n - 1];
                    set_primary_stress(&mut right.pronunciation, target);
                }
            }
            Self::SetSecondaryStressOnLeft => {
                set_all_stress(&mut left.pronunciation, "secondary");
            }
            Self::SetSecondaryStressOnRight => {
                set_all_stress(&mut right.pronunciation, "secondary");
            }
            Self::UnstressLeft => {
                set_all_stress(&mut left.pronunciation, "unstressed");
            }
            Self::UnstressRight => {
                set_all_stress(&mut right.pronunciation, "unstressed");
            }
        }
    }
}

fn is_vowel_token(p: &PhonemeToken) -> bool {
    let major_id = FeatureId("phonology.major".to_string());
    if let Some(Spec::Known(FeatureValue::Category(major))) = p.features.values.get(&major_id) {
        if major == "vowel" {
            return true;
        }
    }
    let syllabic_id = FeatureId("phonology.syllabic".to_string());
    if let Some(Spec::Known(FeatureValue::Bool(syllabic))) = p.features.values.get(&syllabic_id) {
        if *syllabic {
            return true;
        }
    }
    false
}

fn set_primary_stress(pron: &mut [PhonemeToken], target_idx: usize) {
    let stress_id = FeatureId("phonology.stress".to_string());
    for (i, p) in pron.iter_mut().enumerate() {
        if i == target_idx {
            p.features.values.insert(
                stress_id.clone(),
                Spec::Known(FeatureValue::Category("primary".to_string())),
            );
        } else if is_vowel_token(p) {
            if let Some(Spec::Known(FeatureValue::Category(s))) = p.features.values.get(&stress_id)
            {
                if s == "primary" {
                    p.features.values.insert(
                        stress_id.clone(),
                        Spec::Known(FeatureValue::Category("secondary".to_string())),
                    );
                }
            }
        }
    }
}

fn set_all_stress(pron: &mut [PhonemeToken], stress_val: &str) {
    let stress_id = FeatureId("phonology.stress".to_string());
    for p in pron.iter_mut() {
        if is_vowel_token(p) {
            p.features.values.insert(
                stress_id.clone(),
                Spec::Known(FeatureValue::Category(stress_val.to_string())),
            );
        }
    }
}

pub fn compose_morpheme_tokens(
    tokens: &mut [MorphemeToken],
    morpheme_db: &HashMap<MorphemeId, Morpheme>,
    rules: &[MorphologicalRule],
) {
    if tokens.len() < 2 {
        return;
    }

    for i in 0..tokens.len() - 1 {
        let (left_slice, right_slice) = tokens.split_at_mut(i + 1);
        let left = &mut left_slice[i];
        let right = &mut right_slice[0];

        let Spec::Known(left_id) = &left.morpheme else {
            continue;
        };
        let Spec::Known(right_id) = &right.morpheme else {
            continue;
        };

        let dummy_left = Morpheme {
            id: left_id.clone(),
            form: left_id.0.clone(),
            kind: MorphemeKind::Root,
            gloss: None,
            features: FeatureBundle::default(),
            pronunciation: Vec::new(),
        };
        let left_meta = morpheme_db.get(left_id).unwrap_or(&dummy_left);

        let dummy_right = Morpheme {
            id: right_id.clone(),
            form: right_id.0.clone(),
            kind: MorphemeKind::Root,
            gloss: None,
            features: FeatureBundle::default(),
            pronunciation: Vec::new(),
        };
        let right_meta = morpheme_db.get(right_id).unwrap_or(&dummy_right);

        for rule in rules {
            let mut matched = true;
            for trigger in &rule.triggers {
                if !trigger.matches(left, right, left_meta, right_meta) {
                    matched = false;
                    break;
                }
            }
            if matched {
                for action in &rule.actions {
                    action.apply(left, right);
                }
            }
        }
    }
}

pub fn finalize_word_pronunciation(pron: &mut [PhonemeToken]) {
    let stress_id = FeatureId("phonology.stress".to_string());
    let mut primary_idx = None;
    for (i, p) in pron.iter().enumerate() {
        if let Some(Spec::Known(FeatureValue::Category(s))) = p.features.values.get(&stress_id) {
            if s == "primary" {
                primary_idx = Some(i);
            }
        }
    }

    if let Some(primary_idx) = primary_idx {
        for (i, p) in pron.iter_mut().enumerate() {
            if i != primary_idx {
                if let Some(Spec::Known(FeatureValue::Category(s))) =
                    p.features.values.get(&stress_id)
                {
                    if s == "primary" {
                        p.features.values.insert(
                            stress_id.clone(),
                            Spec::Known(FeatureValue::Category("secondary".to_string())),
                        );
                    }
                }
            }
        }
    }
}
