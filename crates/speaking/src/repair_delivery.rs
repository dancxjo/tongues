use serde::{Deserialize, Serialize};

use crate::ids::VarietyId;

pub const REPAIR_DELIVERY_FIXTURE_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairCause {
    Pronunciation,
    Morphology,
    LexicalChoice,
    SentenceBoundary,
    Syntax,
    Meaning,
    AcousticRenderingFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorSource {
    Morphology,
    Syntax,
    Prosody,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairSpan {
    pub start_word: usize,
    pub end_word: usize,
}

impl RepairSpan {
    fn len(self) -> usize {
        self.end_word.saturating_sub(self.start_word)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairAnchor {
    pub word_index: usize,
    pub source: AnchorSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairDetection {
    pub heard_by_listener: bool,
    pub cause: RepairCause,
    pub reparandum: RepairSpan,
    pub repair_span: RepairSpan,
    pub continuation: RepairSpan,
    #[serde(default)]
    pub anchors: Vec<RepairAnchor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairDeliveryMode {
    SilentReplacement,
    CutoffAndRepeat,
    WordPhraseRetrace,
    IMeanReplacement,
    ClauseRestart,
    ApologeticRestart,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "content")]
pub enum Interregnum {
    None,
    Marker(String),
    Apology(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairProsodyPlan {
    pub interruption: bool,
    pub pause_ms: u16,
    pub pitch_reset_semitones: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrastive_prominence: Option<RepairSpan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contour_resumption_at_word: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairProvenance {
    pub reparandum: RepairSpan,
    pub interruption_point_word: usize,
    pub interregnum: Interregnum,
    pub repair_span: RepairSpan,
    pub continuation: RepairSpan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairEvent {
    pub mode: RepairDeliveryMode,
    pub cause: RepairCause,
    pub anchor: RepairAnchor,
    pub prosody: RepairProsodyPlan,
    pub provenance: RepairProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairDeliveryPlan {
    pub variety: VarietyId,
    pub heard_error: bool,
    #[serde(default)]
    pub events: Vec<RepairEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairLexicon {
    pub replacement_markers: Vec<String>,
    pub apology_restarts: Vec<String>,
}

impl RepairLexicon {
    pub fn marker(&self) -> Option<String> {
        self.replacement_markers.first().cloned()
    }

    pub fn apology(&self) -> Option<String> {
        self.apology_restarts.first().cloned()
    }
}

pub fn repair_lexicon_for_variety(variety: &VarietyId) -> RepairLexicon {
    match variety.0.split('-').next().unwrap_or_default() {
        "es" => RepairLexicon {
            replacement_markers: vec!["digo".into()],
            apology_restarts: vec!["perdón".into()],
        },
        "fr" => RepairLexicon {
            replacement_markers: vec!["je veux dire".into()],
            apology_restarts: vec!["pardon".into()],
        },
        _ => RepairLexicon {
            replacement_markers: vec!["I mean".into()],
            apology_restarts: vec!["sorry".into()],
        },
    }
}

pub fn render_repair_plan(variety: VarietyId, detection: &RepairDetection) -> RepairDeliveryPlan {
    if !detection.heard_by_listener {
        return RepairDeliveryPlan {
            variety,
            heard_error: false,
            events: vec![RepairEvent {
                mode: RepairDeliveryMode::SilentReplacement,
                cause: detection.cause,
                anchor: choose_anchor(detection, true),
                prosody: RepairProsodyPlan {
                    interruption: false,
                    pause_ms: 0,
                    pitch_reset_semitones: 0.0,
                    contrastive_prominence: None,
                    contour_resumption_at_word: None,
                },
                provenance: RepairProvenance {
                    reparandum: detection.reparandum,
                    interruption_point_word: detection.reparandum.end_word,
                    interregnum: Interregnum::None,
                    repair_span: detection.repair_span,
                    continuation: detection.continuation,
                },
            }],
        };
    }

    let lexicon = repair_lexicon_for_variety(&variety);
    let mode = choose_mode(detection);
    let interregnum = match mode {
        RepairDeliveryMode::IMeanReplacement => lexicon
            .marker()
            .map(Interregnum::Marker)
            .unwrap_or(Interregnum::None),
        RepairDeliveryMode::ApologeticRestart => lexicon
            .apology()
            .map(Interregnum::Apology)
            .unwrap_or(Interregnum::None),
        _ => Interregnum::None,
    };

    let anchor = choose_anchor(detection, false);
    let contrastive = matches!(
        detection.cause,
        RepairCause::Pronunciation | RepairCause::Morphology | RepairCause::LexicalChoice
    )
    .then_some(detection.repair_span);
    let pause_ms = if mode == RepairDeliveryMode::ClauseRestart {
        220
    } else {
        140
    };

    RepairDeliveryPlan {
        variety,
        heard_error: true,
        events: vec![RepairEvent {
            mode,
            cause: detection.cause,
            anchor,
            prosody: RepairProsodyPlan {
                interruption: true,
                pause_ms,
                pitch_reset_semitones: 1.5,
                contrastive_prominence: contrastive,
                contour_resumption_at_word: Some(detection.continuation.start_word),
            },
            provenance: RepairProvenance {
                reparandum: detection.reparandum,
                interruption_point_word: detection.reparandum.end_word,
                interregnum,
                repair_span: detection.repair_span,
                continuation: detection.continuation,
            },
        }],
    }
}

fn choose_mode(detection: &RepairDetection) -> RepairDeliveryMode {
    match detection.cause {
        RepairCause::Pronunciation if detection.repair_span.len() <= 1 => {
            RepairDeliveryMode::CutoffAndRepeat
        }
        RepairCause::Pronunciation | RepairCause::Morphology => {
            RepairDeliveryMode::WordPhraseRetrace
        }
        RepairCause::LexicalChoice => RepairDeliveryMode::IMeanReplacement,
        RepairCause::SentenceBoundary | RepairCause::Syntax | RepairCause::Meaning => {
            RepairDeliveryMode::ClauseRestart
        }
        RepairCause::AcousticRenderingFailure => RepairDeliveryMode::ApologeticRestart,
    }
}

fn choose_anchor(detection: &RepairDetection, silent: bool) -> RepairAnchor {
    let mut candidates: Vec<_> = detection
        .anchors
        .iter()
        .copied()
        .filter(|anchor| anchor.word_index <= detection.reparandum.start_word)
        .collect();
    if candidates.is_empty() {
        return RepairAnchor {
            word_index: detection.reparandum.start_word,
            source: AnchorSource::Morphology,
        };
    }

    if silent {
        return *candidates
            .iter()
            .max_by_key(|anchor| anchor.word_index)
            .expect("non-empty candidates");
    }

    if matches!(
        detection.cause,
        RepairCause::SentenceBoundary | RepairCause::Syntax | RepairCause::Meaning
    ) {
        candidates.sort_by_key(|anchor| (anchor.word_index, anchor_priority(anchor.source)));
        return *candidates.first().expect("non-empty candidates");
    }

    *candidates
        .iter()
        .max_by_key(|anchor| (anchor.word_index, anchor_priority(anchor.source)))
        .expect("non-empty candidates")
}

fn anchor_priority(source: AnchorSource) -> u8 {
    match source {
        AnchorSource::Syntax => 0,
        AnchorSource::Prosody => 1,
        AnchorSource::Morphology => 2,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairPolicyFixtureSuite {
    pub version: u32,
    pub cases: Vec<RepairPolicyFixture>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairPolicyFixture {
    pub id: String,
    pub variety: VarietyId,
    pub detection: RepairDetection,
    pub expected_mode: RepairDeliveryMode,
    pub expected_anchor_word: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unheard_errors_are_silent_replacement() {
        let plan = render_repair_plan(
            VarietyId("en-US-GA".into()),
            &RepairDetection {
                heard_by_listener: false,
                cause: RepairCause::Pronunciation,
                reparandum: RepairSpan {
                    start_word: 3,
                    end_word: 4,
                },
                repair_span: RepairSpan {
                    start_word: 3,
                    end_word: 4,
                },
                continuation: RepairSpan {
                    start_word: 4,
                    end_word: 7,
                },
                anchors: vec![RepairAnchor {
                    word_index: 3,
                    source: AnchorSource::Morphology,
                }],
            },
        );

        assert_eq!(plan.events.len(), 1);
        assert_eq!(plan.events[0].mode, RepairDeliveryMode::SilentReplacement);
        assert!(!plan.events[0].prosody.interruption);
    }

    #[test]
    fn heard_garden_path_restarts_from_earlier_syntax_anchor() {
        let plan = render_repair_plan(
            VarietyId("en-US-GA".into()),
            &RepairDetection {
                heard_by_listener: true,
                cause: RepairCause::Syntax,
                reparandum: RepairSpan {
                    start_word: 4,
                    end_word: 6,
                },
                repair_span: RepairSpan {
                    start_word: 4,
                    end_word: 7,
                },
                continuation: RepairSpan {
                    start_word: 7,
                    end_word: 10,
                },
                anchors: vec![
                    RepairAnchor {
                        word_index: 0,
                        source: AnchorSource::Syntax,
                    },
                    RepairAnchor {
                        word_index: 3,
                        source: AnchorSource::Morphology,
                    },
                ],
            },
        );

        assert_eq!(plan.events.len(), 1);
        assert_eq!(plan.events[0].mode, RepairDeliveryMode::ClauseRestart);
        assert_eq!(plan.events[0].anchor.word_index, 0);
        assert_eq!(
            plan.events[0].prosody.contour_resumption_at_word,
            Some(7),
            "repair prosody should document contour resumption"
        );
    }

    #[test]
    fn lexicon_markers_are_variety_aware() {
        let en = repair_lexicon_for_variety(&VarietyId("en-US".into()));
        let es = repair_lexicon_for_variety(&VarietyId("es-ES".into()));
        assert_eq!(en.marker().as_deref(), Some("I mean"));
        assert_eq!(es.marker().as_deref(), Some("digo"));
    }

    #[test]
    fn fixture_suite_covers_required_patterns() {
        let suite: RepairPolicyFixtureSuite = serde_json::from_str(include_str!(
            "../../../fixtures/speaking/repair_delivery_policy_v1.json"
        ))
        .expect("fixture suite parses");
        assert_eq!(suite.version, REPAIR_DELIVERY_FIXTURE_VERSION);

        let ids = suite
            .cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for required in [
            "cutoff-and-repeat",
            "repetition-retrace",
            "lexical-replacement",
            "phrase-restart",
            "clause-restart",
        ] {
            assert!(ids.contains(required), "missing fixture {required}");
        }

        for case in suite.cases {
            let plan = render_repair_plan(case.variety, &case.detection);
            let event = plan.events.first().expect("repair event");
            assert_eq!(event.mode, case.expected_mode, "{}", case.id);
            assert_eq!(
                event.anchor.word_index, case.expected_anchor_word,
                "{}",
                case.id
            );
            assert_eq!(
                event.provenance.reparandum, case.detection.reparandum,
                "{}",
                case.id
            );
            assert_eq!(
                event.provenance.repair_span, case.detection.repair_span,
                "{}",
                case.id
            );
            assert_eq!(
                event.provenance.continuation, case.detection.continuation,
                "{}",
                case.id
            );
        }
    }
}
