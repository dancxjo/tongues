//! Provider-neutral language evidence and stable ASR routing policy.

use serde::{Deserialize, Serialize};

use crate::{AudioFrame, LanguageId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedLanguageHypothesis {
    pub language: LanguageId,
    pub confidence: f32,
    pub evidence_ms: u64,
    pub provenance: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageDetection {
    pub segment_id: String,
    pub sequence: u64,
    pub hypotheses: Vec<RankedLanguageHypothesis>,
}

impl LanguageDetection {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.hypotheses.iter().any(|hypothesis| {
            !hypothesis.confidence.is_finite()
                || !(0.0..=1.0).contains(&hypothesis.confidence)
                || hypothesis.language.0.is_empty()
                || hypothesis.provenance.is_empty()
        }) {
            anyhow::bail!(
                "language hypotheses require finite confidence and non-empty identity/provenance"
            );
        }
        if self
            .hypotheses
            .windows(2)
            .any(|pair| pair[0].confidence < pair[1].confidence)
        {
            anyhow::bail!("language hypotheses must be ranked by descending confidence");
        }
        Ok(())
    }
}

pub trait LanguageIdentifier {
    fn identity(&self) -> &str;
    fn detect(
        &mut self,
        segment_id: &str,
        sequence: u64,
        audio: &AudioFrame,
    ) -> anyhow::Result<LanguageDetection>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum LanguageSelectionMode {
    Fixed {
        language: LanguageId,
    },
    Detect {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        candidates: Vec<LanguageId>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LanguageSwitchPolicy {
    pub minimum_confidence: f32,
    pub minimum_evidence_ms: u64,
    pub switch_margin: f32,
    pub consecutive_segments: u32,
}

impl Default for LanguageSwitchPolicy {
    fn default() -> Self {
        Self {
            minimum_confidence: 0.65,
            minimum_evidence_ms: 300,
            switch_margin: 0.15,
            consecutive_segments: 2,
        }
    }
}

impl LanguageSwitchPolicy {
    pub fn validate(self) -> anyhow::Result<Self> {
        if !self.minimum_confidence.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_confidence)
            || !self.switch_margin.is_finite()
            || !(0.0..=1.0).contains(&self.switch_margin)
            || self.consecutive_segments == 0
        {
            anyhow::bail!("invalid language switching policy");
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsrLanguageCapability {
    pub provider_id: String,
    pub model_id: String,
    pub installed: bool,
    pub languages: Vec<LanguageId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageIdentifierCapability {
    pub detector_id: String,
    pub model_id: String,
    pub installed: bool,
    pub languages: Vec<LanguageId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageRoutingCapabilities {
    pub selection_modes: Vec<String>,
    pub default_switch_policy: LanguageSwitchPolicy,
    pub detectors: Vec<LanguageIdentifierCapability>,
    pub asr_providers: Vec<AsrLanguageCapability>,
    pub unsupported_language_policies: Vec<String>,
}

impl LanguageRoutingCapabilities {
    pub fn new(
        detectors: Vec<LanguageIdentifierCapability>,
        asr_providers: Vec<AsrLanguageCapability>,
    ) -> Self {
        Self {
            selection_modes: vec!["fixed".into(), "detect".into()],
            default_switch_policy: LanguageSwitchPolicy::default(),
            detectors,
            asr_providers,
            unsupported_language_policies: vec!["error".into(), "fallback".into()],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "policy")]
pub enum UnsupportedLanguagePolicy {
    Error,
    Fallback {
        provider_id: String,
        language: LanguageId,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageRoute {
    /// Detection remains visible even when the selected route falls back.
    pub detection: Option<LanguageDetection>,
    pub segment_sequence: u64,
    pub selected_language: LanguageId,
    pub provider_id: String,
    pub model_id: String,
    pub changed_language: bool,
    pub fallback_reason: Option<String>,
}

pub struct LanguageRouter {
    mode: LanguageSelectionMode,
    switching: LanguageSwitchPolicy,
    unsupported: UnsupportedLanguagePolicy,
    providers: Vec<AsrLanguageCapability>,
    active: Option<LanguageId>,
    pending: Option<(LanguageId, u32)>,
    last_sequence: Option<u64>,
}

impl LanguageRouter {
    pub fn new(
        mode: LanguageSelectionMode,
        switching: LanguageSwitchPolicy,
        unsupported: UnsupportedLanguagePolicy,
        providers: Vec<AsrLanguageCapability>,
    ) -> anyhow::Result<Self> {
        let switching = switching.validate()?;
        if providers.iter().any(|provider| {
            provider.provider_id.is_empty()
                || provider.model_id.is_empty()
                || provider.languages.is_empty()
        }) {
            anyhow::bail!("ASR language capabilities require provider, model, and languages");
        }
        let active = match &mode {
            LanguageSelectionMode::Fixed { language } => Some(language.clone()),
            LanguageSelectionMode::Detect { .. } => None,
        };
        Ok(Self {
            mode,
            switching,
            unsupported,
            providers,
            active,
            pending: None,
            last_sequence: None,
        })
    }

    pub fn capabilities(&self) -> &[AsrLanguageCapability] {
        &self.providers
    }

    pub fn route(
        &mut self,
        segment_sequence: u64,
        detection: Option<LanguageDetection>,
    ) -> anyhow::Result<LanguageRoute> {
        if self
            .last_sequence
            .is_some_and(|previous| segment_sequence <= previous)
        {
            anyhow::bail!("language routing received an out-of-order segment");
        }
        if detection
            .as_ref()
            .is_some_and(|detection| detection.sequence != segment_sequence)
        {
            anyhow::bail!("language detection sequence does not match the routed segment");
        }
        if let Some(detection) = &detection {
            detection.validate()?;
        }
        let previous = self.active.clone();
        let mode = self.mode.clone();
        let selected = match mode {
            LanguageSelectionMode::Fixed { language } => language.clone(),
            LanguageSelectionMode::Detect { candidates } => {
                self.detected_selection(detection.as_ref(), &candidates)?
            }
        };
        self.active = Some(selected.clone());
        self.last_sequence = Some(segment_sequence);
        let changed_language = previous.as_ref().is_some_and(|old| old != &selected);

        if let Some(provider) = self.compatible_provider(&selected) {
            return Ok(LanguageRoute {
                detection,
                segment_sequence,
                selected_language: selected,
                provider_id: provider.provider_id.clone(),
                model_id: provider.model_id.clone(),
                changed_language,
                fallback_reason: None,
            });
        }
        match &self.unsupported {
            UnsupportedLanguagePolicy::Error => {
                anyhow::bail!(
                    "no installed ASR provider supports language `{}`",
                    selected.0
                )
            }
            UnsupportedLanguagePolicy::Fallback {
                provider_id,
                language,
            } => {
                let provider = self
                    .providers
                    .iter()
                    .find(|provider| {
                        provider.installed
                            && &provider.provider_id == provider_id
                            && provider.languages.contains(language)
                    })
                    .ok_or_else(|| {
                        anyhow::anyhow!("configured language fallback is unavailable")
                    })?;
                Ok(LanguageRoute {
                    detection,
                    segment_sequence,
                    selected_language: language.clone(),
                    provider_id: provider.provider_id.clone(),
                    model_id: provider.model_id.clone(),
                    changed_language,
                    fallback_reason: Some(format!(
                        "detected language `{}` is unsupported; routed as `{}`",
                        selected.0, language.0
                    )),
                })
            }
        }
    }

    fn detected_selection(
        &mut self,
        detection: Option<&LanguageDetection>,
        candidates: &[LanguageId],
    ) -> anyhow::Result<LanguageId> {
        let top = detection
            .and_then(|detection| {
                detection.hypotheses.iter().find(|hypothesis| {
                    candidates.is_empty() || candidates.contains(&hypothesis.language)
                })
            })
            .filter(|hypothesis| {
                hypothesis.confidence >= self.switching.minimum_confidence
                    && hypothesis.evidence_ms >= self.switching.minimum_evidence_ms
            });
        let Some(top) = top else {
            return self
                .active
                .clone()
                .ok_or_else(|| anyhow::anyhow!("language evidence is ambiguous or insufficient"));
        };
        let Some(active) = &self.active else {
            self.pending = None;
            return Ok(top.language.clone());
        };
        if active == &top.language {
            self.pending = None;
            return Ok(active.clone());
        }
        let active_confidence = detection
            .and_then(|detection| {
                detection
                    .hypotheses
                    .iter()
                    .find(|hypothesis| &hypothesis.language == active)
            })
            .map_or(0.0, |hypothesis| hypothesis.confidence);
        if top.confidence < active_confidence + self.switching.switch_margin {
            self.pending = None;
            return Ok(active.clone());
        }
        let count = self
            .pending
            .as_ref()
            .filter(|(language, _)| language == &top.language)
            .map_or(1, |(_, count)| count.saturating_add(1));
        self.pending = Some((top.language.clone(), count));
        if count < self.switching.consecutive_segments {
            return Ok(active.clone());
        }
        self.pending = None;
        Ok(top.language.clone())
    }

    fn compatible_provider(&self, language: &LanguageId) -> Option<&AsrLanguageCapability> {
        self.providers
            .iter()
            .find(|provider| provider.installed && provider.languages.contains(language))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn language(value: &str) -> LanguageId {
        LanguageId(value.into())
    }

    fn detection(segment: usize, ranked: &[(&str, f32)]) -> LanguageDetection {
        LanguageDetection {
            segment_id: format!("segment:{segment}"),
            sequence: segment as u64,
            hypotheses: ranked
                .iter()
                .map(|(id, confidence)| RankedLanguageHypothesis {
                    language: language(id),
                    confidence: *confidence,
                    evidence_ms: 500,
                    provenance: "fixture-lid".into(),
                })
                .collect(),
        }
    }

    fn providers() -> Vec<AsrLanguageCapability> {
        vec![
            AsrLanguageCapability {
                provider_id: "english-asr".into(),
                model_id: "en-v1".into(),
                installed: true,
                languages: vec![language("en")],
            },
            AsrLanguageCapability {
                provider_id: "spanish-asr".into(),
                model_id: "es-v1".into(),
                installed: true,
                languages: vec![language("es")],
            },
        ]
    }

    #[test]
    fn fixed_language_routes_without_detection() {
        let mut router = LanguageRouter::new(
            LanguageSelectionMode::Fixed {
                language: language("en"),
            },
            LanguageSwitchPolicy::default(),
            UnsupportedLanguagePolicy::Error,
            providers(),
        )
        .unwrap();
        assert_eq!(router.route(0, None).unwrap().provider_id, "english-asr");
    }

    #[test]
    fn mixed_language_requires_stable_consecutive_evidence() {
        let mut router = LanguageRouter::new(
            LanguageSelectionMode::Detect { candidates: vec![] },
            LanguageSwitchPolicy::default(),
            UnsupportedLanguagePolicy::Error,
            providers(),
        )
        .unwrap();
        assert_eq!(
            router
                .route(0, Some(detection(0, &[("en", 0.9), ("es", 0.1)])))
                .unwrap()
                .selected_language,
            language("en")
        );
        assert_eq!(
            router
                .route(1, Some(detection(1, &[("es", 0.9), ("en", 0.1)])))
                .unwrap()
                .selected_language,
            language("en")
        );
        let switched = router
            .route(2, Some(detection(2, &[("es", 0.92), ("en", 0.08)])))
            .unwrap();
        assert_eq!(switched.selected_language, language("es"));
        assert!(switched.changed_language);
    }

    #[test]
    fn ambiguous_speech_does_not_flap_active_language() {
        let mut router = LanguageRouter::new(
            LanguageSelectionMode::Detect { candidates: vec![] },
            LanguageSwitchPolicy::default(),
            UnsupportedLanguagePolicy::Error,
            providers(),
        )
        .unwrap();
        router.route(0, Some(detection(0, &[("en", 0.9)]))).unwrap();
        let route = router
            .route(1, Some(detection(1, &[("es", 0.55), ("en", 0.45)])))
            .unwrap();
        assert_eq!(route.selected_language, language("en"));
        assert_eq!(
            route.detection.unwrap().hypotheses[0].language,
            language("es")
        );
    }

    #[test]
    fn fallback_keeps_detected_evidence_visible() {
        let mut router = LanguageRouter::new(
            LanguageSelectionMode::Detect { candidates: vec![] },
            LanguageSwitchPolicy {
                consecutive_segments: 1,
                ..LanguageSwitchPolicy::default()
            },
            UnsupportedLanguagePolicy::Fallback {
                provider_id: "english-asr".into(),
                language: language("en"),
            },
            providers(),
        )
        .unwrap();
        let route = router
            .route(0, Some(detection(0, &[("fr", 0.95)])))
            .unwrap();
        assert_eq!(route.selected_language, language("en"));
        assert_eq!(
            route.detection.unwrap().hypotheses[0].language,
            language("fr")
        );
        assert!(route.fallback_reason.is_some());
    }

    #[test]
    fn segment_routing_rejects_reordering() {
        let mut router = LanguageRouter::new(
            LanguageSelectionMode::Fixed {
                language: language("en"),
            },
            LanguageSwitchPolicy::default(),
            UnsupportedLanguagePolicy::Error,
            providers(),
        )
        .unwrap();
        router.route(4, None).unwrap();
        assert!(
            router
                .route(3, None)
                .unwrap_err()
                .to_string()
                .contains("out-of-order")
        );
    }
}
