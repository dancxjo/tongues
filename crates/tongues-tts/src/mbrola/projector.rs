use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use speaking::{
    phone_display_symbol, BoundaryKind, Curve, FeatureValue, PhoneToken, ProsodicLabelKind, Spec,
    Stress, TimeSpan, UtterancePlan,
};
use thiserror::Error;

use super::{MbrolaPhone, MbrolaPitchTarget, PhoneTimedPlan};

pub const MBROLA_SILENCE: &str = "_";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MbrolaVoiceMetadata {
    pub id: String,
    pub variety: String,
    pub baseline_hz: Option<f32>,
    pub pitch_range_hz: Option<f32>,
}

/// Runtime configuration for a logical voice backed by an MBROLA diphone
/// database. More than one logical voice may deliberately share a database
/// while selecting a different variety and symbol projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MbrolaVoiceConfig {
    pub id: &'static str,
    pub display_name: &'static str,
    pub database_id: &'static str,
    pub database_voice_id: &'static str,
    pub variety: &'static str,
    pub symbol_map_id: &'static str,
    pub baseline_hz: Option<f32>,
    pub pitch_range_hz: Option<f32>,
}

pub const MBROLA_VOICE_CONFIGS: &[MbrolaVoiceConfig] = &[
    MbrolaVoiceConfig {
        id: "mbrola-us1",
        display_name: "MBROLA US English us1",
        database_id: "mbrola-us1",
        database_voice_id: "us1",
        variety: "en-US",
        symbol_map_id: "us1",
        baseline_hz: None,
        pitch_range_hz: None,
    },
    MbrolaVoiceConfig {
        id: "mbrola-us3",
        display_name: "MBROLA US English us3",
        database_id: "mbrola-us3",
        database_voice_id: "us3",
        variety: "en-US",
        symbol_map_id: "us3",
        baseline_hz: None,
        pitch_range_hz: None,
    },
    MbrolaVoiceConfig {
        id: "mbrola-en1",
        display_name: "MBROLA British English en1",
        database_id: "mbrola-en1",
        database_voice_id: "en1",
        variety: "en-GB",
        symbol_map_id: "en1",
        baseline_hz: None,
        pitch_range_hz: None,
    },
    MbrolaVoiceConfig {
        id: "mbrola-nl2",
        display_name: "MBROLA Dutch nl2",
        database_id: "mbrola-nl2",
        database_voice_id: "nl2",
        variety: "nl-NL",
        symbol_map_id: "nl2",
        baseline_hz: None,
        pitch_range_hz: None,
    },
    MbrolaVoiceConfig {
        id: "mbrola-eo-nl2",
        display_name: "Esperanto via MBROLA Dutch nl2",
        database_id: "mbrola-nl2",
        database_voice_id: "nl2",
        variety: "eo",
        symbol_map_id: "eo-nl2",
        baseline_hz: None,
        pitch_range_hz: None,
    },
    MbrolaVoiceConfig {
        id: "mbrola-la-la1",
        display_name: "Classical Latin via MBROLA la1",
        database_id: "mbrola-la1",
        database_voice_id: "la1",
        variety: "la-Classical",
        symbol_map_id: "la-la1",
        baseline_hz: None,
        pitch_range_hz: None,
    },
    MbrolaVoiceConfig {
        id: "mbrola-sa-in1",
        display_name: "Sanskrit via MBROLA Hindi in1",
        database_id: "mbrola-in1",
        database_voice_id: "in1",
        variety: "sa-Deva-Standard",
        symbol_map_id: "sa-in1",
        baseline_hz: None,
        pitch_range_hz: None,
    },
    MbrolaVoiceConfig {
        id: "mbrola-sa-in2",
        display_name: "Sanskrit via MBROLA Hindi in2",
        database_id: "mbrola-in2",
        database_voice_id: "in2",
        variety: "sa-Deva-Standard",
        symbol_map_id: "sa-in2",
        baseline_hz: None,
        pitch_range_hz: None,
    },
];

impl MbrolaVoiceConfig {
    pub fn for_id(id: &str) -> Option<&'static Self> {
        MBROLA_VOICE_CONFIGS.iter().find(|config| config.id == id)
    }

    pub fn for_database_and_variety(
        database_voice_id: &str,
        variety: &str,
    ) -> Option<&'static Self> {
        MBROLA_VOICE_CONFIGS.iter().find(|config| {
            config.database_voice_id == database_voice_id && config.variety == variety
        })
    }

    pub fn symbol_map(&self) -> MbrolaSymbolMap {
        MbrolaSymbolMap::for_voice_id(self.symbol_map_id)
            .expect("registered MBROLA voice configuration must have a symbol map")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MbrolaTimingProfile {
    pub consonant_ms: u32,
    pub vowel_ms: u32,
    pub stressed_nucleus_scale: f32,
    pub focused_nucleus_scale: f32,
    pub word_break_ms: u32,
    pub phrase_break_ms: u32,
    pub breath_group_break_ms: u32,
    pub turn_break_ms: u32,
    pub min_break_ms: u32,
    pub max_break_ms: u32,
    pub pitch_samples_per_phone: usize,
    pub fallback_baseline_hz: f32,
    pub fallback_pitch_range_hz: f32,
}

impl Default for MbrolaTimingProfile {
    fn default() -> Self {
        Self {
            consonant_ms: 72,
            vowel_ms: 110,
            stressed_nucleus_scale: 1.25,
            focused_nucleus_scale: 1.15,
            word_break_ms: 20,
            phrase_break_ms: 110,
            breath_group_break_ms: 180,
            turn_break_ms: 240,
            min_break_ms: 20,
            max_break_ms: 800,
            pitch_samples_per_phone: 3,
            fallback_baseline_hz: 120.0,
            fallback_pitch_range_hz: 30.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MbrolaSymbolMap {
    pub id: String,
    pub mappings: BTreeMap<String, String>,
    #[serde(default)]
    pub expansions: BTreeMap<String, Vec<String>>,
}

impl MbrolaSymbolMap {
    pub fn new(
        id: impl Into<String>,
        mappings: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self {
            id: id.into(),
            mappings: mappings
                .into_iter()
                .map(|(from, to)| (from.into(), to.into()))
                .collect(),
            expansions: BTreeMap::new(),
        }
    }

    pub fn with_expansions(
        mut self,
        expansions: impl IntoIterator<Item = (impl Into<String>, Vec<impl Into<String>>)>,
    ) -> Self {
        self.expansions = expansions
            .into_iter()
            .map(|(from, to)| (from.into(), to.into_iter().map(Into::into).collect()))
            .collect();
        self
    }

    pub fn identity(id: impl Into<String>) -> Self {
        Self::new(id, std::iter::empty::<(String, String)>())
    }

    /// Returns the built-in map for the voice databases distributed by the
    /// upstream MBROLA voice catalog.
    pub fn for_voice_id(voice_id: &str) -> Option<Self> {
        let common = [
            ("_", "_"),
            ("pau", "_"),
            ("sil", "_"),
            ("h", "h"),
            ("l", "l"),
            ("ɫ", "l"),
            ("m", "m"),
            ("b", "b"),
            ("ɪ", "I"),
            ("æ", "{"),
            ("ɡ", "g"),
            ("ʊ", "U"),
            ("ʌ", "V"),
            ("ə", "@"),
            ("aʊ", "aU"),
            ("ɔɪ", "OI"),
            ("p", "p"),
            ("pʰ", "p"),
            ("p˭", "p"),
            ("t", "t"),
            ("tʰ", "t"),
            ("t˭", "t"),
            ("k", "k"),
            ("kʰ", "k"),
            ("k˭", "k"),
            ("d", "d"),
            ("g", "g"),
            ("n", "n"),
            ("ŋ", "N"),
            ("f", "f"),
            ("v", "v"),
            ("θ", "T"),
            ("ð", "D"),
            ("s", "s"),
            ("z", "z"),
            ("ʃ", "S"),
            ("ʒ", "Z"),
            ("r", "r"),
            ("ɹ", "r"),
            ("w", "w"),
            ("j", "j"),
            ("tʃ", "tS"),
            ("dʒ", "dZ"),
            ("HH", "h"),
            ("AE", "{"),
            ("AH", "@"),
            ("AW", "aU"),
            ("OY", "OI"),
            ("UH", "U"),
            ("AH0", "@"),
            ("AH1", "V"),
            ("AE0", "{"),
            ("AE1", "{"),
            ("L", "l"),
            ("M", "m"),
            ("N", "n"),
            ("NG", "N"),
            ("B", "b"),
            ("P", "p"),
            ("T", "t"),
            ("D", "d"),
            ("K", "k"),
            ("G", "g"),
            ("F", "f"),
            ("V", "v"),
            ("TH", "T"),
            ("DH", "D"),
            ("S", "s"),
            ("Z", "z"),
            ("SH", "S"),
            ("ZH", "Z"),
            ("R", "r"),
            ("W", "w"),
            ("Y", "j"),
            ("CH", "tS"),
            ("JH", "dZ"),
        ];
        let american = [
            ("i", "i"),
            ("iː", "i"),
            ("ɛ", "E"),
            ("ɑ", "A"),
            ("ɔ", "O"),
            ("u", "u"),
            ("uː", "u"),
            ("AA", "A"),
            ("AO", "O"),
            ("EH", "E"),
            ("IH", "I"),
            ("IY", "i"),
            ("UW", "u"),
            ("AA1", "A"),
            ("AO1", "O"),
            ("EH1", "E"),
            ("IH1", "I"),
            ("IY0", "i"),
            ("IY1", "i"),
            ("UW1", "u"),
        ];
        let variants: &[(&str, &str)] = match voice_id {
            "us1" | "mbrola-us1" => &[
                ("ɝ", "3"),
                ("ɚ", "3"),
                ("oʊ", "oU"),
                ("aɪ", "aI"),
                ("ɑɪ", "aI"),
                ("eɪ", "eI"),
                ("ER", "3"),
                ("OW", "oU"),
                ("AY", "aI"),
                ("EY", "eI"),
                ("ER0", "3"),
                ("ER1", "3"),
                ("OW0", "oU"),
                ("OW1", "oU"),
                ("AY0", "aI"),
                ("AY1", "aI"),
                ("EY0", "eI"),
                ("EY1", "eI"),
                ("DX", "d"),
            ],
            "us3" | "mbrola-us3" => &[
                ("ɝ", "r="),
                ("ɚ", "r="),
                ("oʊ", "@U"),
                ("aɪ", "AI"),
                ("ɑɪ", "AI"),
                ("eɪ", "EI"),
                ("ɾ", "4"),
                ("ER", "r="),
                ("OW", "@U"),
                ("AY", "AI"),
                ("EY", "EI"),
                ("ER0", "r="),
                ("ER1", "r="),
                ("OW0", "@U"),
                ("OW1", "@U"),
                ("AY0", "AI"),
                ("AY1", "AI"),
                ("EY0", "EI"),
                ("EY1", "EI"),
                ("DX", "4"),
            ],
            "en1" | "mbrola-en1" => {
                return Some(Self::new(
                    "mbrola-en1-built-in",
                    common.into_iter().chain([
                        ("i", "i:"),
                        ("iː", "i:"),
                        ("e", "e"),
                        ("ɛ", "e"),
                        ("ɑ", "A:"),
                        ("ɑː", "A:"),
                        ("ɒ", "Q"),
                        ("ɔ", "O:"),
                        ("ɔː", "O:"),
                        ("u", "u:"),
                        ("uː", "u:"),
                        ("ɜ", "3:"),
                        ("ɜː", "3:"),
                        ("ɝ", "3:"),
                        ("ɚ", "3:"),
                        ("oʊ", "@U"),
                        ("əʊ", "@U"),
                        ("aɪ", "aI"),
                        ("ɑɪ", "aI"),
                        ("eɪ", "eI"),
                        ("eə", "e@"),
                        ("ɪə", "I@"),
                        ("ʊə", "U@"),
                        ("AA", "A:"),
                        ("AO", "O:"),
                        ("EH", "e"),
                        ("ER", "3:"),
                        ("EY", "eI"),
                        ("IH", "I"),
                        ("IY", "i:"),
                        ("OW", "@U"),
                        ("AY", "aI"),
                        ("UW", "u:"),
                        ("AA0", "A:"),
                        ("AA1", "A:"),
                        ("AO0", "O:"),
                        ("AO1", "O:"),
                        ("EH0", "e"),
                        ("EH1", "e"),
                        ("ER0", "3:"),
                        ("ER1", "3:"),
                        ("EY0", "eI"),
                        ("EY1", "eI"),
                        ("IH0", "I"),
                        ("IH1", "I"),
                        ("IY0", "i:"),
                        ("IY1", "i:"),
                        ("OW0", "@U"),
                        ("OW1", "@U"),
                        ("AY0", "aI"),
                        ("AY1", "aI"),
                        ("UW0", "u:"),
                        ("UW1", "u:"),
                        ("DX", "d"),
                    ]),
                ));
            }
            "nl2" | "mbrola-nl2" => {
                return Some(Self::new(
                    "mbrola-nl2-built-in",
                    [
                        ("_", "_"),
                        ("a", "a"),
                        ("b", "b"),
                        ("d", "d"),
                        ("e", "e"),
                        ("f", "f"),
                        ("g", "g"),
                        ("ɡ", "g"),
                        ("h", "h"),
                        ("i", "i"),
                        ("j", "j"),
                        ("k", "k"),
                        ("l", "l"),
                        ("m", "m"),
                        ("n", "n"),
                        ("o", "o"),
                        ("p", "p"),
                        ("r", "r"),
                        ("s", "s"),
                        ("t", "t"),
                        ("u", "u"),
                        ("v", "v"),
                        ("w", "w"),
                        ("x", "x"),
                        ("z", "z"),
                        ("ʃ", "S"),
                        ("ʒ", "Z"),
                    ],
                ));
            }
            "eo-nl2" | "mbrola-eo-nl2" => {
                return Some(
                    Self::new(
                        "mbrola-eo-nl2-built-in",
                        [
                            ("_", "_"),
                            ("a", "a"),
                            ("b", "b"),
                            ("d", "d"),
                            ("e", "e"),
                            ("f", "f"),
                            ("g", "g"),
                            ("ɡ", "g"),
                            ("h", "h"),
                            ("i", "i"),
                            ("j", "j"),
                            ("k", "k"),
                            ("l", "l"),
                            ("m", "m"),
                            ("n", "n"),
                            ("o", "o"),
                            ("p", "p"),
                            ("r", "r"),
                            ("s", "s"),
                            ("t", "t"),
                            ("u", "u"),
                            ("v", "v"),
                            ("w", "w"),
                            ("x", "x"),
                            ("z", "z"),
                            ("ʃ", "S"),
                            ("ʒ", "Z"),
                        ],
                    )
                    .with_expansions([
                        ("t͡s", vec!["t", "s"]),
                        ("t͡ʃ", vec!["t", "S"]),
                        ("d͡ʒ", vec!["d", "Z"]),
                    ]),
                );
            }
            "la-la1" | "mbrola-la-la1" => {
                return Some(
                    Self::new(
                        "mbrola-la-la1-built-in",
                        [
                            ("_", "_"),
                            ("a", "a"),
                            ("e", "E"),
                            ("i", "I"),
                            ("o", "O"),
                            ("u", "U"),
                            ("y", "y"),
                            ("ae̯", "aE"),
                            ("au̯", "aU"),
                            ("oe̯", "OE"),
                            ("b", "b"),
                            ("k", "k"),
                            ("kʰ", "k_h"),
                            ("d", "d"),
                            ("f", "f"),
                            ("ɡ", "g"),
                            ("h", "h"),
                            ("j", "j"),
                            ("l", "l"),
                            ("m", "m"),
                            ("n", "n"),
                            ("p", "p"),
                            ("pʰ", "p_h"),
                            ("r", "r"),
                            ("s", "s"),
                            ("t", "t"),
                            ("tʰ", "t_h"),
                            ("w", "w"),
                            ("z", "z"),
                        ],
                    )
                    .with_expansions([("ks", vec!["k", "s"])]),
                );
            }
            "sa-in1" | "mbrola-sa-in1" | "sa-in2" | "mbrola-sa-in2" => {
                let database = if voice_id.ends_with("in2") {
                    "in2"
                } else {
                    "in1"
                };
                return Some(
                    Self::new(
                        format!("mbrola-sa-{database}-built-in"),
                        [
                            ("_", "_"),
                            ("a", "a"),
                            ("aː", "aa"),
                            ("i", "ii"),
                            ("iː", "ii"),
                            ("u", "uu"),
                            ("uː", "uu"),
                            ("eː", "e"),
                            ("ai̯", "ai"),
                            ("oː", "o"),
                            ("au̯", "au"),
                            ("k", "k"),
                            ("kʰ", "kh"),
                            ("ɡ", "g"),
                            ("ɡʱ", "gh"),
                            ("ŋ", "n"),
                            ("t͡ɕ", "c"),
                            ("t͡ɕʰ", "ch"),
                            ("d͡ʑ", "j"),
                            ("d͡ʑʱ", "jh"),
                            ("ɲ", "n"),
                            ("ʈ", "T"),
                            ("ʈʰ", "Th"),
                            ("ɖ", "D"),
                            ("ɖʱ", "Dh"),
                            ("ɳ", "N"),
                            ("t", "t"),
                            ("tʰ", "th"),
                            ("d", "d"),
                            ("dʱ", "dh"),
                            ("n", "n"),
                            ("p", "p"),
                            ("pʰ", "ph"),
                            ("b", "b"),
                            ("bʱ", "bh"),
                            ("m", "m"),
                            ("j", "y"),
                            ("r", "r"),
                            ("l", "l"),
                            ("v", "v"),
                            ("ɕ", "sh"),
                            ("ʂ", "sh"),
                            ("s", "s"),
                            ("ɦ", "h"),
                            ("h", "h"),
                        ],
                    )
                    // Hindi MBROLA has no syllabic-r unit. Preserve the
                    // rhotic onset and supply a short vocalic release.
                    .with_expansions([("r̩", vec!["r", "ii"])]),
                );
            }
            _ => return None,
        };
        Some(Self::new(
            format!("mbrola-{voice_id}-built-in"),
            common
                .into_iter()
                .chain(american)
                .chain(variants.iter().copied()),
        ))
    }

    fn resolve(
        &self,
        phone: &str,
        voice: &MbrolaVoiceMetadata,
        inventory: &BTreeSet<String>,
    ) -> Result<Vec<String>, MbrolaLoweringError> {
        let stressless = phone.trim_end_matches(|ch: char| ch.is_ascii_digit());
        if let Some(expansion) = self
            .expansions
            .get(phone)
            .or_else(|| self.expansions.get(stressless))
        {
            for mapped in expansion {
                if !inventory.contains(mapped) {
                    return Err(MbrolaLoweringError::UnsupportedVoiceSymbol {
                        phone: phone.to_string(),
                        mapped: mapped.clone(),
                        variety: voice.variety.clone(),
                        voice: voice.id.clone(),
                        symbol_map: self.id.clone(),
                    });
                }
            }
            return Ok(expansion.clone());
        }
        let mapped = self
            .mappings
            .get(phone)
            .or_else(|| self.mappings.get(stressless))
            .cloned()
            .or_else(|| self.mappings.is_empty().then(|| stressless.to_string()))
            .ok_or_else(|| MbrolaLoweringError::UnknownPhoneMapping {
                phone: phone.to_string(),
                variety: voice.variety.clone(),
                voice: voice.id.clone(),
                symbol_map: self.id.clone(),
            })?;
        if !inventory.contains(&mapped) {
            return Err(MbrolaLoweringError::UnsupportedVoiceSymbol {
                phone: phone.to_string(),
                mapped,
                variety: voice.variety.clone(),
                voice: voice.id.clone(),
                symbol_map: self.id.clone(),
            });
        }
        Ok(vec![mapped])
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MbrolaLoweringReport {
    pub voice: String,
    pub symbol_map: String,
    pub explicit_span_phones: usize,
    pub inferred_duration_phones: usize,
    pub explicit_pitch_phones: usize,
    pub fallback_pitch_phones: usize,
    pub inserted_breaks: usize,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MbrolaProjector {
    pub voice: MbrolaVoiceMetadata,
    pub symbol_map: MbrolaSymbolMap,
    pub inventory: BTreeSet<String>,
    pub timing: MbrolaTimingProfile,
    pub control_baseline_hz: Option<f32>,
    pub control_pitch_range_hz: Option<f32>,
}

impl MbrolaProjector {
    pub fn project(
        &self,
        plan: &UtterancePlan,
    ) -> Result<(PhoneTimedPlan, MbrolaLoweringReport), MbrolaLoweringError> {
        if plan.target_phones.is_empty() {
            return Err(MbrolaLoweringError::EmptyPlan);
        }
        let baseline = self
            .control_baseline_hz
            .or(self.voice.baseline_hz)
            .unwrap_or(self.timing.fallback_baseline_hz);
        let pitch_range = self
            .control_pitch_range_hz
            .or(self.voice.pitch_range_hz)
            .unwrap_or(self.timing.fallback_pitch_range_hz);
        if !baseline.is_finite() || baseline <= 0.0 || !pitch_range.is_finite() {
            return Err(MbrolaLoweringError::InvalidPitchProfile);
        }

        let mut phones = Vec::new();
        let mut report = MbrolaLoweringReport {
            voice: self.voice.id.clone(),
            symbol_map: self.symbol_map.id.clone(),
            explicit_span_phones: 0,
            inferred_duration_phones: 0,
            explicit_pitch_phones: 0,
            fallback_pitch_phones: 0,
            inserted_breaks: 0,
            limitations: Vec::new(),
        };
        let mut inferred_cursor_s = 0.0;
        let mut breaks = plan.target_prosody.breaks.iter().collect::<Vec<_>>();
        breaks.sort_by(|left, right| left.after_s.total_cmp(&right.after_s));
        let mut break_index = 0;
        let speech_boundaries = speech_boundary_insertions(plan, &self.timing);
        let mut speech_boundary_index = 0;
        let mut previous_single_symbol: Option<String> = None;

        for (index, token) in plan.target_phones.iter().enumerate() {
            while speech_boundary_index < speech_boundaries.len()
                && speech_boundaries[speech_boundary_index].0 == index
            {
                phones.push(MbrolaPhone::new(
                    MBROLA_SILENCE,
                    speech_boundaries[speech_boundary_index].1,
                ));
                report.inserted_breaks += 1;
                speech_boundary_index += 1;
                previous_single_symbol = None;
            }
            let phone = spec_phone(token)?;
            if is_structural_boundary(phone) {
                if !matches!(phone.as_str(), "ipa.phone.|" | "boundary.word") {
                    previous_single_symbol = None;
                }
                continue;
            }
            let source = phone_display_symbol(phone);
            let symbols = self
                .symbol_map
                .resolve(source, &self.voice, &self.inventory)?;
            let syllable = syllable_metadata(plan, index);
            let span = token.span.unwrap_or_else(|| {
                let duration = inferred_duration_s(
                    token,
                    syllable.as_ref(),
                    inferred_cursor_s,
                    plan,
                    &self.timing,
                );
                TimeSpan {
                    start_s: inferred_cursor_s,
                    end_s: inferred_cursor_s + duration,
                }
            });
            if token.span.is_some() {
                report.explicit_span_phones += 1;
            } else {
                report.inferred_duration_phones += 1;
            }
            if !span.start_s.is_finite() || !span.end_s.is_finite() || span.end_s <= span.start_s {
                return Err(MbrolaLoweringError::InvalidPhoneSpan {
                    phone: source.to_string(),
                    start_s: span.start_s,
                    end_s: span.end_s,
                });
            }

            while break_index < breaks.len() && breaks[break_index].after_s <= span.start_s {
                let duration_ms = break_duration_ms(breaks[break_index], &self.timing)?;
                phones.push(MbrolaPhone::new(MBROLA_SILENCE, duration_ms));
                report.inserted_breaks += 1;
                break_index += 1;
                previous_single_symbol = None;
            }

            let mut duration_ms = seconds_to_ms(span.duration_s())?;
            if token.span.is_none() {
                if syllable
                    .as_ref()
                    .is_some_and(|metadata| metadata.stressed_nucleus)
                {
                    duration_ms = scaled_duration(duration_ms, self.timing.stressed_nucleus_scale);
                }
                if label_intersects(plan, &ProsodicLabelKind::Focus, span)
                    || label_intersects(plan, &ProsodicLabelKind::Emphasis, span)
                {
                    duration_ms = scaled_duration(duration_ms, self.timing.focused_nucleus_scale);
                }
            }
            let voiced = is_voiced(token, source);
            let explicit_targets = if voiced {
                sample_explicit_pitch(
                    &plan.target_prosody.pitch,
                    span,
                    self.timing.pitch_samples_per_phone,
                )?
            } else {
                Vec::new()
            };
            let pitch_targets = if !explicit_targets.is_empty() {
                report.explicit_pitch_phones += 1;
                explicit_targets
            } else if voiced {
                report.fallback_pitch_phones += 1;
                fallback_targets(plan, span, syllable.as_ref(), baseline, pitch_range)
            } else {
                Vec::new()
            };
            let split_durations = split_duration(duration_ms, symbols.len())?;
            let symbol_count = split_durations.len();
            let geminate = (symbols.len() == 1 && !is_vowel_symbol(source))
                .then(|| format!("{}:", symbols[0]))
                .filter(|geminate| self.inventory.contains(geminate))
                .filter(|_| previous_single_symbol.as_deref() == Some(symbols[0].as_str()));
            if let Some(geminate) = geminate {
                let previous = phones
                    .last_mut()
                    .expect("a previous single symbol exists for geminate lowering");
                previous.symbol = geminate;
                previous.duration_ms = previous
                    .duration_ms
                    .checked_add(duration_ms)
                    .ok_or(MbrolaLoweringError::DurationOverflow)?;
                previous_single_symbol = None;
            } else {
                for (symbol_index, (symbol, symbol_duration)) in
                    symbols.into_iter().zip(split_durations).enumerate()
                {
                    let single_symbol = (symbol_count == 1).then(|| symbol.clone());
                    phones.push(
                        MbrolaPhone::new(symbol, symbol_duration).with_pitch_targets(
                            split_pitch_targets(&pitch_targets, symbol_index, symbol_count),
                        ),
                    );
                    previous_single_symbol = single_symbol;
                }
            }
            inferred_cursor_s = span.end_s;
        }

        while break_index < breaks.len() {
            phones.push(MbrolaPhone::new(
                MBROLA_SILENCE,
                break_duration_ms(breaks[break_index], &self.timing)?,
            ));
            report.inserted_breaks += 1;
            break_index += 1;
        }
        while speech_boundary_index < speech_boundaries.len() {
            phones.push(MbrolaPhone::new(
                MBROLA_SILENCE,
                speech_boundaries[speech_boundary_index].1,
            ));
            report.inserted_breaks += 1;
            speech_boundary_index += 1;
        }
        if !plan.target_prosody.energy.points.is_empty() || plan.style.is_some() {
            report.limitations.push(
                ".pho cannot encode energy or style; these remain on the UtterancePlan".into(),
            );
        }
        Ok((PhoneTimedPlan::new(phones), report))
    }
}

fn is_structural_boundary(phone: &speaking::PhoneId) -> bool {
    phone.as_str().starts_with("boundary.") || phone.as_str() == "ipa.phone.|"
}

fn split_duration(duration_ms: u32, parts: usize) -> Result<Vec<u32>, MbrolaLoweringError> {
    if parts == 0 || duration_ms < parts as u32 {
        return Err(MbrolaLoweringError::InvalidExpansionDuration { duration_ms, parts });
    }
    let base = duration_ms / parts as u32;
    let remainder = duration_ms % parts as u32;
    Ok((0..parts)
        .map(|index| base + u32::from(index < remainder as usize))
        .collect())
}

fn split_pitch_targets(
    targets: &[MbrolaPitchTarget],
    part: usize,
    parts: usize,
) -> Vec<MbrolaPitchTarget> {
    if parts <= 1 {
        return targets.to_vec();
    }
    let lower = part as f32 / parts as f32;
    let upper = (part + 1) as f32 / parts as f32;
    targets
        .iter()
        .filter_map(|target| {
            let global = target.percent as f32 / 100.0;
            let is_last_endpoint = part + 1 == parts && target.percent == 100;
            ((global >= lower && global < upper) || is_last_endpoint).then(|| MbrolaPitchTarget {
                percent: (((global - lower) / (upper - lower)) * 100.0)
                    .round()
                    .clamp(0.0, 100.0) as u8,
                hz: target.hz,
            })
        })
        .collect()
}

fn speech_boundary_insertions(
    plan: &UtterancePlan,
    profile: &MbrolaTimingProfile,
) -> Vec<(usize, u32)> {
    let word_boundary_indices = plan
        .target_phones
        .iter()
        .enumerate()
        .filter_map(|(index, token)| {
            spec_phone(token)
                .ok()
                .is_some_and(|phone| phone.as_str() == "boundary.word")
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let mut boundaries = plan
        .boundaries
        .iter()
        .filter_map(|boundary| {
            let duration = match (&boundary.kind, boundary.pause, boundary.terminal) {
                (_, Some(_), _) => profile.phrase_break_ms,
                (BoundaryKind::Phrase, _, _) => profile.phrase_break_ms,
                (BoundaryKind::BreathGroup, _, _) => profile.breath_group_break_ms,
                (BoundaryKind::Turn, _, _) | (_, _, Some(_)) => profile.turn_break_ms,
                // An ordinary word boundary is not a pause. Keeping adjacent
                // phones contiguous lets MBROLA use the cross-word diphone
                // instead of writing a short run of zero-valued samples.
                (BoundaryKind::Word, _, _) => return None,
                _ => return None,
            }
            .clamp(profile.min_break_ms, profile.max_break_ms);
            // Phonemicizer boundaries use the completed word's index. Place
            // the pause at the corresponding structural word token rather
            // than proportionally projecting that index across characters
            // and phones, which can put silence inside a word.
            let phone_index = word_boundary_indices
                .get(boundary.after_grapheme_index)
                .copied()
                .unwrap_or(plan.target_phones.len());
            Some((phone_index, duration))
        })
        .collect::<Vec<_>>();
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

#[derive(Debug, Clone)]
struct SyllableMetadata {
    stressed_nucleus: bool,
}

fn syllable_metadata(plan: &UtterancePlan, phone_index: usize) -> Option<SyllableMetadata> {
    let mut cursor = 0;
    for syllable in &plan.target_syllables {
        let end = cursor + syllable.phones.len();
        if phone_index < end {
            let local = phone_index - cursor;
            let stressed = matches!(
                syllable.stress,
                Spec::Known(Stress::Primary)
                    | Spec::Known(Stress::Secondary)
                    | Spec::Gradient {
                        value: Stress::Primary | Stress::Secondary,
                        ..
                    }
            );
            return Some(SyllableMetadata {
                stressed_nucleus: stressed && syllable.nucleus_index == Some(local),
            });
        }
        cursor = end;
    }
    None
}

fn spec_phone(token: &PhoneToken) -> Result<&speaking::PhoneId, MbrolaLoweringError> {
    match &token.phone {
        Spec::Known(phone) | Spec::Gradient { value: phone, .. } => Ok(phone),
        Spec::Variable(values) if values.len() == 1 => Ok(&values[0]),
        Spec::Variable(values) => Err(MbrolaLoweringError::AmbiguousPhone {
            alternatives: values
                .iter()
                .map(|value| value.as_str().to_string())
                .collect(),
        }),
        _ => Err(MbrolaLoweringError::UnspecifiedPhone),
    }
}

fn inferred_duration_s(
    token: &PhoneToken,
    _syllable: Option<&SyllableMetadata>,
    time_s: f64,
    plan: &UtterancePlan,
    profile: &MbrolaTimingProfile,
) -> f64 {
    let source = spec_phone(token)
        .map(phone_display_symbol)
        .unwrap_or_default();
    let base_ms = if is_vowel_symbol(source) {
        if is_long_vowel_symbol(source) {
            scaled_duration(profile.vowel_ms, 2.0)
        } else {
            profile.vowel_ms
        }
    } else {
        profile.consonant_ms
    };
    let rate = curve_value_at(&plan.target_prosody.speaking_rate, time_s)
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .unwrap_or(1.0);
    f64::from(base_ms) / f64::from(rate) / 1000.0
}

fn break_duration_ms(
    item: &speaking::ProsodicBreak,
    profile: &MbrolaTimingProfile,
) -> Result<u32, MbrolaLoweringError> {
    let configured = match item.boundary {
        BoundaryKind::Word => profile.word_break_ms,
        BoundaryKind::Phrase => profile.phrase_break_ms,
        BoundaryKind::BreathGroup => profile.breath_group_break_ms,
        BoundaryKind::Turn => profile.turn_break_ms,
        _ => profile.min_break_ms,
    };
    let requested = match item.duration_s {
        Spec::Known(value) | Spec::Gradient { value, .. } => {
            if !value.is_finite() || value <= 0.0 {
                return Err(MbrolaLoweringError::InvalidBreakDuration(value));
            }
            (value * 1000.0).round() as u32
        }
        _ => configured,
    };
    Ok(requested.clamp(profile.min_break_ms, profile.max_break_ms))
}

fn sample_explicit_pitch(
    curve: &Curve,
    span: TimeSpan,
    samples: usize,
) -> Result<Vec<MbrolaPitchTarget>, MbrolaLoweringError> {
    if curve.points.is_empty() {
        return Ok(Vec::new());
    }
    let samples = samples.max(2);
    let mut targets = Vec::new();
    for index in 0..samples {
        let fraction = index as f64 / (samples - 1) as f64;
        let time_s = span.start_s + span.duration_s() * fraction;
        if let Some(hz) = curve_value_at(curve, time_s) {
            if !hz.is_finite() || hz <= 0.0 {
                return Err(MbrolaLoweringError::InvalidPitchPoint { time_s, hz });
            }
            targets.push(MbrolaPitchTarget {
                percent: (fraction * 100.0).round() as u8,
                hz,
            });
        }
    }
    Ok(targets)
}

fn curve_value_at(curve: &Curve, time_s: f64) -> Option<f32> {
    let mut points = curve
        .points
        .iter()
        .filter(|point| point.time_s.is_finite() && point.value.is_finite())
        .collect::<Vec<_>>();
    points.sort_by(|left, right| left.time_s.total_cmp(&right.time_s));
    let first = *points.first()?;
    if time_s <= first.time_s {
        return Some(first.value);
    }
    for pair in points.windows(2) {
        if time_s <= pair[1].time_s {
            let width = pair[1].time_s - pair[0].time_s;
            if width <= f64::EPSILON {
                return Some(pair[1].value);
            }
            let fraction = ((time_s - pair[0].time_s) / width).clamp(0.0, 1.0) as f32;
            return Some(pair[0].value * (1.0 - fraction) + pair[1].value * fraction);
        }
    }
    points.last().map(|point| point.value)
}

fn fallback_targets(
    plan: &UtterancePlan,
    span: TimeSpan,
    syllable: Option<&SyllableMetadata>,
    baseline: f32,
    range: f32,
) -> Vec<MbrolaPitchTarget> {
    let prominent = syllable.is_some_and(|metadata| metadata.stressed_nucleus)
        || label_intersects(plan, &ProsodicLabelKind::Focus, span)
        || label_intersects(plan, &ProsodicLabelKind::Emphasis, span);
    let start = baseline + if prominent { range * 0.25 } else { 0.0 };
    let mut end = start;
    for label in &plan.target_prosody.labels {
        if intersects(label.span, span) {
            match label.kind {
                ProsodicLabelKind::QuestionRise
                | ProsodicLabelKind::AlternativeQuestionRise
                | ProsodicLabelKind::ContinuationRise => end = baseline + range,
                ProsodicLabelKind::FinalFall | ProsodicLabelKind::AlternativeQuestionFall => {
                    end = (baseline - range * 0.6).max(40.0)
                }
                _ => {}
            }
        }
    }
    vec![
        MbrolaPitchTarget {
            percent: 0,
            hz: start,
        },
        MbrolaPitchTarget {
            percent: 100,
            hz: end,
        },
    ]
}

fn label_intersects(plan: &UtterancePlan, kind: &ProsodicLabelKind, span: TimeSpan) -> bool {
    plan.target_prosody
        .labels
        .iter()
        .any(|label| &label.kind == kind && intersects(label.span, span))
}

fn intersects(left: TimeSpan, right: TimeSpan) -> bool {
    left.start_s < right.end_s && right.start_s < left.end_s
}

fn is_voiced(token: &PhoneToken, symbol: &str) -> bool {
    for (feature, value) in &token.features.values {
        if feature.0.ends_with("voicing") || feature.0 == "voiced" {
            match value {
                Spec::Known(FeatureValue::Bool(value))
                | Spec::Gradient {
                    value: FeatureValue::Bool(value),
                    ..
                } => return *value,
                Spec::Known(FeatureValue::Category(value))
                | Spec::Gradient {
                    value: FeatureValue::Category(value),
                    ..
                } => return value != "voiceless",
                _ => {}
            }
        }
    }
    is_vowel_symbol(symbol)
        || matches!(
            symbol.trim_end_matches(|ch: char| ch.is_ascii_digit()),
            "m" | "n"
                | "ng"
                | "ŋ"
                | "l"
                | "r"
                | "w"
                | "j"
                | "v"
                | "z"
                | "zh"
                | "ʒ"
                | "b"
                | "d"
                | "g"
                | "dh"
                | "ð"
        )
}

fn is_vowel_symbol(symbol: &str) -> bool {
    let symbol = symbol.trim_end_matches(|ch: char| ch.is_ascii_digit());
    matches!(
        symbol,
        "a" | "e"
            | "i"
            | "o"
            | "u"
            | "aː"
            | "eː"
            | "iː"
            | "oː"
            | "uː"
            | "r̩"
            | "ae̯"
            | "ai̯"
            | "au̯"
            | "oe̯"
            | "ə"
            | "ɚ"
            | "ɝ"
            | "æ"
            | "ɑ"
            | "ɔ"
            | "ɛ"
            | "ɪ"
            | "ʊ"
            | "ʌ"
            | "AH"
            | "AA"
            | "AE"
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

fn is_long_vowel_symbol(symbol: &str) -> bool {
    let symbol = symbol.trim_end_matches(|ch: char| ch.is_ascii_digit());
    symbol.contains('ː') || matches!(symbol, "ae̯" | "ai̯" | "au̯" | "oe̯")
}

fn seconds_to_ms(seconds: f64) -> Result<u32, MbrolaLoweringError> {
    let milliseconds = seconds * 1000.0;
    if !milliseconds.is_finite() || milliseconds <= 0.0 || milliseconds > u32::MAX as f64 {
        return Err(MbrolaLoweringError::InvalidDuration(seconds));
    }
    Ok(milliseconds.round().max(1.0) as u32)
}

fn scaled_duration(duration: u32, scale: f32) -> u32 {
    ((duration as f32 * scale).round() as u32).max(1)
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum MbrolaLoweringError {
    #[error("cannot lower an UtterancePlan with no target phones")]
    EmptyPlan,
    #[error("target phone is unspecified")]
    UnspecifiedPhone,
    #[error("target phone is ambiguous: {alternatives:?}")]
    AmbiguousPhone { alternatives: Vec<String> },
    #[error("phone `{phone}` in variety `{variety}` has no mapping for voice `{voice}` in symbol map `{symbol_map}`")]
    UnknownPhoneMapping {
        phone: String,
        variety: String,
        voice: String,
        symbol_map: String,
    },
    #[error("phone `{phone}` maps to unsupported symbol `{mapped}` for voice `{voice}` (variety `{variety}`, symbol map `{symbol_map}`)")]
    UnsupportedVoiceSymbol {
        phone: String,
        mapped: String,
        variety: String,
        voice: String,
        symbol_map: String,
    },
    #[error("phone `{phone}` has invalid span {start_s}..{end_s} seconds")]
    InvalidPhoneSpan {
        phone: String,
        start_s: f64,
        end_s: f64,
    },
    #[error("invalid duration {0} seconds")]
    InvalidDuration(f64),
    #[error("cannot split {duration_ms} ms across {parts} MBROLA symbols")]
    InvalidExpansionDuration { duration_ms: u32, parts: usize },
    #[error("MBROLA phone duration overflow while combining a geminate")]
    DurationOverflow,
    #[error("invalid prosodic break duration {0} seconds")]
    InvalidBreakDuration(f32),
    #[error("pitch point at {time_s} seconds must be finite and positive, got {hz}")]
    InvalidPitchPoint { time_s: f64, hz: f32 },
    #[error("MBROLA voice pitch baseline/range is invalid")]
    InvalidPitchProfile,
}

#[cfg(test)]
mod tests {
    use speaking::{
        CurvePoint, EvidenceProvenance, EvidenceSource, FeatureBundle, PhoneId, ProsodicBreak,
        ProsodicLabel, ProsodyTrack, Syllable, UtteranceId, VarietyId,
    };

    use super::*;

    fn token(symbol: &'static str, span: Option<TimeSpan>) -> PhoneToken {
        PhoneToken {
            phone: Spec::Known(PhoneId::borrowed(symbol)),
            span,
            features: FeatureBundle::default(),
            acoustic_evidence: Vec::new(),
            confidence: 1.0,
            provenance: EvidenceProvenance {
                source: EvidenceSource::TtsPlan,
                method: "fixture".into(),
                version: None,
            },
        }
    }

    fn plan() -> UtterancePlan {
        let phones = vec![
            token(
                "h",
                Some(TimeSpan {
                    start_s: 0.0,
                    end_s: 0.08,
                }),
            ),
            token(
                "AH1",
                Some(TimeSpan {
                    start_s: 0.08,
                    end_s: 0.20,
                }),
            ),
        ];
        UtterancePlan {
            id: UtteranceId("fixture".into()),
            variety: VarietyId("en-US".into()),
            speaker: None,
            intended_text: Some("huh?".into()),
            intended_morphemes: Vec::new(),
            intended_phonemes: Vec::new(),
            target_phones: phones.clone(),
            target_syllables: vec![Syllable {
                phones,
                stress: Spec::Known(Stress::Primary),
                phone_positions: Vec::new(),
                span: Some(TimeSpan {
                    start_s: 0.0,
                    end_s: 0.20,
                }),
                nucleus_index: Some(1),
            }],
            boundaries: Vec::new(),
            target_prosody: ProsodyTrack {
                pitch: Curve {
                    points: vec![
                        CurvePoint {
                            time_s: 0.08,
                            value: 110.0,
                            confidence: 1.0,
                        },
                        CurvePoint {
                            time_s: 0.20,
                            value: 150.0,
                            confidence: 1.0,
                        },
                    ],
                },
                breaks: vec![ProsodicBreak {
                    after_s: 0.20,
                    duration_s: Spec::Known(0.15),
                    boundary: BoundaryKind::Phrase,
                    confidence: 1.0,
                }],
                labels: vec![ProsodicLabel {
                    span: TimeSpan {
                        start_s: 0.08,
                        end_s: 0.20,
                    },
                    kind: ProsodicLabelKind::QuestionRise,
                    confidence: 1.0,
                }],
                ..ProsodyTrack::default()
            },
            target_acoustics: Vec::new(),
            speaker_reference: None,
            style: None,
            provenance: EvidenceProvenance {
                source: EvidenceSource::TtsPlan,
                method: "fixture".into(),
                version: None,
            },
        }
    }

    #[test]
    fn explicit_spans_pitch_and_break_lower_deterministically() {
        let projector = MbrolaProjector {
            voice: MbrolaVoiceMetadata {
                id: "tiny".into(),
                variety: "en-US".into(),
                baseline_hz: Some(115.0),
                pitch_range_hz: Some(35.0),
            },
            symbol_map: MbrolaSymbolMap::new(
                "fixture-map",
                [("h", "h"), ("AH", "@"), ("AH1", "@")],
            ),
            inventory: ["_", "h", "@"].into_iter().map(str::to_string).collect(),
            timing: MbrolaTimingProfile::default(),
            control_baseline_hz: None,
            control_pitch_range_hz: None,
        };
        let (lowered, report) = projector.project(&plan()).unwrap();
        assert_eq!(lowered.phones[0], MbrolaPhone::new("h", 80));
        assert_eq!(lowered.phones[1].symbol, "@");
        assert_eq!(lowered.phones[1].duration_ms, 120);
        assert_eq!(lowered.phones[1].pitch_targets[0].hz, 110.0);
        assert_eq!(lowered.phones[1].pitch_targets[2].hz, 150.0);
        assert_eq!(lowered.phones[2], MbrolaPhone::new("_", 150));
        assert_eq!(report.explicit_pitch_phones, 1);
        assert_eq!(report.inserted_breaks, 1);
    }

    #[test]
    fn fallback_duration_and_contour_are_stable_and_unvoiced_has_no_f0() {
        let mut plan = plan();
        plan.target_phones
            .iter_mut()
            .for_each(|phone| phone.span = None);
        plan.target_syllables.iter_mut().for_each(|syllable| {
            syllable
                .phones
                .iter_mut()
                .for_each(|phone| phone.span = None)
        });
        plan.target_prosody.pitch.points.clear();
        plan.target_prosody.breaks.clear();
        let projector = MbrolaProjector {
            voice: MbrolaVoiceMetadata {
                id: "tiny".into(),
                variety: "en-US".into(),
                baseline_hz: Some(100.0),
                pitch_range_hz: Some(40.0),
            },
            symbol_map: MbrolaSymbolMap::new("fixture-map", [("h", "h"), ("AH1", "@")]),
            inventory: ["_", "h", "@"].into_iter().map(str::to_string).collect(),
            timing: MbrolaTimingProfile::default(),
            control_baseline_hz: None,
            control_pitch_range_hz: None,
        };
        let (first, _) = projector.project(&plan).unwrap();
        let (second, _) = projector.project(&plan).unwrap();
        assert_eq!(first, second);
        assert!(first.phones[0].pitch_targets.is_empty());
        assert!(first.phones[1].duration_ms > first.phones[0].duration_ms);
        assert!(first.phones[1].pitch_targets.last().unwrap().hz > 100.0);
    }

    #[test]
    fn english_en1_lowers_ipa_allophones_to_card_inventory() {
        let config = MbrolaVoiceConfig::for_id("mbrola-en1").expect("British English voice config");
        let voice = MbrolaVoiceMetadata {
            id: config.id.into(),
            variety: config.variety.into(),
            baseline_hz: None,
            pitch_range_hz: None,
        };
        let inventory = ["_", "k", "l", "p", "t"]
            .into_iter()
            .map(str::to_string)
            .collect();

        let map = config.symbol_map();
        for (phone, expected) in [
            ("ɫ", "l"),
            ("pʰ", "p"),
            ("p˭", "p"),
            ("tʰ", "t"),
            ("t˭", "t"),
            ("kʰ", "k"),
            ("k˭", "k"),
        ] {
            assert_eq!(
                map.resolve(phone, &voice, &inventory).unwrap(),
                vec![expected],
                "en1 card projection for {phone}"
            );
        }
    }

    #[test]
    fn american_voices_lower_english_stop_allophones_to_card_inventory() {
        let inventory = ["_", "k", "p", "t"]
            .into_iter()
            .map(str::to_string)
            .collect();
        for id in ["mbrola-us1", "mbrola-us3"] {
            let config = MbrolaVoiceConfig::for_id(id).expect("American English voice config");
            let voice = MbrolaVoiceMetadata {
                id: config.id.into(),
                variety: config.variety.into(),
                baseline_hz: None,
                pitch_range_hz: None,
            };
            let map = config.symbol_map();
            for (phone, expected) in [
                ("pʰ", "p"),
                ("p˭", "p"),
                ("tʰ", "t"),
                ("t˭", "t"),
                ("kʰ", "k"),
                ("k˭", "k"),
            ] {
                assert_eq!(
                    map.resolve(phone, &voice, &inventory).unwrap(),
                    vec![expected],
                    "{id} card projection for {phone}"
                );
            }
        }
    }

    #[test]
    fn structural_phone_boundaries_are_not_lowered_as_voice_symbols() {
        let mut plan = plan();
        plan.target_phones.insert(1, token("boundary.word", None));
        plan.target_phones.insert(2, token("ipa.phone.|", None));
        let projector = MbrolaProjector {
            voice: MbrolaVoiceMetadata {
                id: "tiny".into(),
                variety: "en-GB".into(),
                baseline_hz: Some(115.0),
                pitch_range_hz: Some(35.0),
            },
            symbol_map: MbrolaSymbolMap::new(
                "fixture-map",
                [("h", "h"), ("AH", "@"), ("AH1", "@")],
            ),
            inventory: ["_", "h", "@"].into_iter().map(str::to_string).collect(),
            timing: MbrolaTimingProfile::default(),
            control_baseline_hz: None,
            control_pitch_range_hz: None,
        };

        let (lowered, _) = projector.project(&plan).unwrap();

        assert_eq!(
            lowered
                .phones
                .iter()
                .map(|phone| phone.symbol.as_str())
                .collect::<Vec<_>>(),
            vec!["h", "@", "_"]
        );
    }

    #[test]
    fn ordinary_word_boundaries_do_not_insert_silence() {
        let mut plan = plan();
        plan.intended_text = Some("huh huh".into());
        plan.boundaries.push(speaking::SpeechBoundaryToken {
            kind: BoundaryKind::Word,
            after_grapheme_index: 0,
            span: None,
            terminal: None,
            pause: None,
        });

        assert!(speech_boundary_insertions(&plan, &MbrolaTimingProfile::default()).is_empty());
    }

    #[test]
    fn paused_boundaries_land_on_structural_word_tokens() {
        let mut plan = plan();
        plan.intended_text = Some("a deliberately long first word, huh".into());
        plan.target_phones.insert(1, token("boundary.word", None));
        plan.boundaries.push(speaking::SpeechBoundaryToken {
            kind: BoundaryKind::Word,
            after_grapheme_index: 0,
            span: None,
            terminal: None,
            pause: Some(speaking::PauseKind::Comma),
        });

        assert_eq!(
            speech_boundary_insertions(&plan, &MbrolaTimingProfile::default()),
            vec![(1, 110)]
        );
    }

    #[test]
    fn esperanto_nl2_configuration_covers_the_complete_inventory() {
        let config = MbrolaVoiceConfig::for_id("mbrola-eo-nl2").expect("Esperanto voice config");
        assert_eq!(config.database_id, "mbrola-nl2");
        assert_eq!(config.variety, "eo");
        let map = config.symbol_map();
        let inventory = [
            "_", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "r",
            "s", "t", "u", "v", "w", "x", "z", "S", "Z",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        let voice = MbrolaVoiceMetadata {
            id: config.id.into(),
            variety: config.variety.into(),
            baseline_hz: None,
            pitch_range_hz: None,
        };
        for phone in [
            "a", "b", "t͡s", "t͡ʃ", "d", "e", "f", "ɡ", "d͡ʒ", "h", "x", "i", "j", "ʒ", "k", "l", "m",
            "n", "o", "p", "r", "s", "ʃ", "t", "u", "w", "v", "z",
        ] {
            map.resolve(phone, &voice, &inventory)
                .unwrap_or_else(|error| panic!("Esperanto phone {phone} is not covered: {error}"));
        }
        assert_eq!(
            map.resolve("t͡s", &voice, &inventory).unwrap(),
            vec!["t", "s"]
        );
        assert_eq!(
            map.resolve("t͡ʃ", &voice, &inventory).unwrap(),
            vec!["t", "S"]
        );
        assert_eq!(
            map.resolve("d͡ʒ", &voice, &inventory).unwrap(),
            vec!["d", "Z"]
        );
    }

    #[test]
    fn latin_la1_configuration_covers_the_classical_inventory() {
        let config = MbrolaVoiceConfig::for_id("mbrola-la-la1").expect("Latin voice config");
        assert_eq!(config.database_id, "mbrola-la1");
        assert_eq!(config.variety, "la-Classical");
        let map = config.symbol_map();
        let inventory = [
            "_", "E", "I", "O", "OE", "U", "a", "aE", "aU", "b", "d", "f", "g", "h", "j", "k",
            "k_h", "l", "m", "n", "p", "p_h", "r", "s", "t", "t_h", "w", "y", "z",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        let voice = MbrolaVoiceMetadata {
            id: config.id.into(),
            variety: config.variety.into(),
            baseline_hz: None,
            pitch_range_hz: None,
        };
        for phone in [
            "a", "e", "i", "o", "u", "y", "ae̯", "au̯", "oe̯", "b", "k", "kʰ", "d", "f", "ɡ", "h",
            "j", "l", "m", "n", "p", "pʰ", "r", "s", "t", "tʰ", "w", "ks", "z",
        ] {
            map.resolve(phone, &voice, &inventory)
                .unwrap_or_else(|error| panic!("Latin phone {phone} is not covered: {error}"));
        }
        assert_eq!(
            map.resolve("ks", &voice, &inventory).unwrap(),
            vec!["k", "s"]
        );
    }

    #[test]
    fn latin_la1_uses_card_geminate_units_for_doubled_consonants() {
        let config = MbrolaVoiceConfig::for_id("mbrola-la-la1").expect("Latin voice config");
        let mut plan = plan();
        plan.variety = VarietyId(config.variety.into());
        plan.target_phones = vec![
            token("s", None),
            token("ipa.phone.|", None),
            token("s", None),
        ];
        plan.target_syllables.clear();
        plan.boundaries.clear();
        plan.target_prosody = ProsodyTrack::default();
        let projector = MbrolaProjector {
            voice: MbrolaVoiceMetadata {
                id: config.id.into(),
                variety: config.variety.into(),
                baseline_hz: None,
                pitch_range_hz: None,
            },
            symbol_map: config.symbol_map(),
            inventory: ["_", "s", "s:"].into_iter().map(str::to_string).collect(),
            timing: MbrolaTimingProfile::default(),
            control_baseline_hz: None,
            control_pitch_range_hz: None,
        };

        let (lowered, _) = projector.project(&plan).unwrap();

        assert_eq!(lowered.phones, vec![MbrolaPhone::new("s:", 144)]);
    }

    #[test]
    fn sanskrit_hindi_configurations_cover_the_tongues_inventory() {
        let inventory = [
            "_", "D", "Dh", "N", "T", "Th", "a", "aa", "ai", "au", "b", "bh", "c", "ch", "d", "dh",
            "e", "g", "gh", "h", "ii", "j", "jh", "k", "kh", "l", "m", "n", "o", "p", "ph", "r",
            "s", "sh", "t", "th", "uu", "v", "y",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        for id in ["mbrola-sa-in1", "mbrola-sa-in2"] {
            let config = MbrolaVoiceConfig::for_id(id).expect("Sanskrit voice config");
            assert_eq!(config.variety, "sa-Deva-Standard");
            let map = config.symbol_map();
            let voice = MbrolaVoiceMetadata {
                id: config.id.into(),
                variety: config.variety.into(),
                baseline_hz: None,
                pitch_range_hz: None,
            };
            for phone in [
                "a", "aː", "i", "iː", "u", "uː", "r̩", "eː", "ai̯", "oː", "au̯", "k", "kʰ", "ɡ", "ɡʱ",
                "ŋ", "t͡ɕ", "t͡ɕʰ", "d͡ʑ", "d͡ʑʱ", "ɲ", "ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ", "t", "tʰ", "d",
                "dʱ", "n", "p", "pʰ", "b", "bʱ", "m", "j", "r", "l", "v", "ɕ", "ʂ", "s", "ɦ", "h",
            ] {
                map.resolve(phone, &voice, &inventory)
                    .unwrap_or_else(|error| {
                        panic!("Sanskrit phone {phone} is not covered by {id}: {error}")
                    });
            }
            assert_eq!(
                map.resolve("r̩", &voice, &inventory).unwrap(),
                vec!["r", "ii"]
            );
        }
    }

    #[test]
    fn ipa_long_vowels_and_diphthongs_receive_vowel_fallback_timing() {
        let plan = plan();
        let profile = MbrolaTimingProfile::default();
        let short = inferred_duration_s(&token("i", None), None, 0.0, &plan, &profile);
        let long = inferred_duration_s(&token("iː", None), None, 0.0, &plan, &profile);
        let diphthong = inferred_duration_s(&token("ai̯", None), None, 0.0, &plan, &profile);
        let syllabic_r = inferred_duration_s(&token("r̩", None), None, 0.0, &plan, &profile);

        assert_eq!(short, 0.110);
        assert_eq!(long, 0.220);
        assert_eq!(diphthong, 0.220);
        assert_eq!(syllabic_r, 0.110);
    }
}
