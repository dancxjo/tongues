use std::collections::HashMap;

use crate::feature::FeatureSystem;
use crate::ids::{LanguageId, PhoneId, PhonemeId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::rules::{PhonotacticConstraint, Phonotactics, RuleStatus, SyllableShape};
use crate::segment::{Environment, SegmentMatcher, SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::variety::{LinguisticVariety, VarietyImplementationStatus, VarietyStatus};

const A: PhoneId = PhoneId::borrowed("ipa.phone.a");
const E: PhoneId = PhoneId::borrowed("ipa.phone.e");
const I: PhoneId = PhoneId::borrowed("ipa.phone.i");
const O: PhoneId = PhoneId::borrowed("ipa.phone.o");
const U: PhoneId = PhoneId::borrowed("ipa.phone.u");
const P: PhoneId = PhoneId::borrowed("ipa.phone.p");
const L: PhoneId = PhoneId::borrowed("ipa.phone.l");
const R: PhoneId = PhoneId::borrowed("ipa.phone.r");
const S: PhoneId = PhoneId::borrowed("ipa.phone.s");
const N: PhoneId = PhoneId::borrowed("ipa.phone.n");
const M: PhoneId = PhoneId::borrowed("ipa.phone.m");
const T: PhoneId = PhoneId::borrowed("ipa.phone.t");
const K: PhoneId = PhoneId::borrowed("ipa.phone.k");

#[derive(Debug, Clone, PartialEq, Eq)]
struct EsperantoSegment {
    symbol: &'static str,
    phone: PhoneId,
}

const PHONEMES: &[EsperantoSegment] = &[
    EsperantoSegment {
        symbol: "A",
        phone: A,
    },
    EsperantoSegment {
        symbol: "E",
        phone: E,
    },
    EsperantoSegment {
        symbol: "I",
        phone: I,
    },
    EsperantoSegment {
        symbol: "O",
        phone: O,
    },
    EsperantoSegment {
        symbol: "U",
        phone: U,
    },
    EsperantoSegment {
        symbol: "P",
        phone: P,
    },
    EsperantoSegment {
        symbol: "L",
        phone: L,
    },
    EsperantoSegment {
        symbol: "R",
        phone: R,
    },
    EsperantoSegment {
        symbol: "S",
        phone: S,
    },
    EsperantoSegment {
        symbol: "N",
        phone: N,
    },
    EsperantoSegment {
        symbol: "M",
        phone: M,
    },
    EsperantoSegment {
        symbol: "T",
        phone: T,
    },
    EsperantoSegment {
        symbol: "K",
        phone: K,
    },
];

const ONSET_CLUSTERS: &[&[PhoneId]] = &[&[P, L], &[P, R]];

pub fn variety() -> LinguisticVariety {
    let mut phonemes = HashMap::new();
    let mut phones = HashMap::new();
    for segment in PHONEMES {
        let phone_id = segment.phone.clone();
        let ipa = phone_symbol(&phone_id);
        phones.insert(
            phone_id.clone(),
            Phone {
                id: phone_id.clone(),
                ipa: ipa.into(),
                features: Default::default(),
                aliases: vec![SymbolAlias {
                    system: "esperanto".into(),
                    symbol: segment.symbol.into(),
                }],
                status: SegmentStatus::Core,
            },
        );
        let phoneme = Phoneme {
            id: PhonemeId(format!("eo.phoneme.{ipa}")),
            notation: format!("/{ipa}/"),
            features: Default::default(),
            default_phone: Some(phone_id.clone()),
            possible_phones: vec![phone_id],
            aliases: vec![SymbolAlias {
                system: "esperanto".into(),
                symbol: segment.symbol.into(),
            }],
            allophones: Vec::new(),
            status: SegmentStatus::Core,
        };
        phonemes.insert(phoneme.id.clone(), phoneme);
    }

    LinguisticVariety {
        id: VarietyId("eo".into()),
        language: LanguageId("eo".into()),
        name: "Esperanto (sample)".into(),
        feature_system: FeatureSystem::default(),
        phonemes: PhonemeInventory { phonemes },
        phones: PhoneInventory { phones },
        allophone_rules: Vec::new(),
        epenthesis_rules: Vec::new(),
        weak_forms: Vec::new(),
        orthographic_unit_pronunciations: Vec::new(),
        phonotactics: Some(Phonotactics {
            allowed_syllable_shapes: vec![
                SyllableShape {
                    pattern: "V".into(),
                },
                SyllableShape {
                    pattern: "CV".into(),
                },
                SyllableShape {
                    pattern: "CVC".into(),
                },
            ],
            constraints: ONSET_CLUSTERS
                .iter()
                .map(|cluster| cluster_constraint(cluster))
                .collect(),
        }),
        orthography: Some(Orthography {
            name: "Esperanto Latin orthography".into(),
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: None,
        status: VarietyStatus::Attested,
        implementation_status: VarietyImplementationStatus::Complete,
    }
}

fn cluster_constraint(cluster: &[PhoneId]) -> PhonotacticConstraint {
    let suffix = cluster_suffix(cluster);
    let label = cluster_label(cluster);
    PhonotacticConstraint {
        id: format!("eo.legal_onset.{suffix}"),
        description: format!("Legal Esperanto onset cluster {label}"),
        matcher: SegmentMatcher::Any,
        environment: Environment {
            before: cluster.iter().cloned().map(SegmentMatcher::Phone).collect(),
            syllable_position: Spec::Known(crate::segment::SyllablePosition::Onset),
            ..Default::default()
        },
        status: RuleStatus::Productive,
    }
}

fn cluster_suffix(cluster: &[PhoneId]) -> String {
    cluster
        .iter()
        .map(phone_symbol)
        .collect::<Vec<_>>()
        .join("_")
}

fn cluster_label(cluster: &[PhoneId]) -> String {
    cluster
        .iter()
        .map(phone_symbol)
        .collect::<Vec<_>>()
        .join("")
}

fn phone_symbol(phone: &PhoneId) -> &str {
    phone
        .as_str()
        .strip_prefix("ipa.phone.")
        .unwrap_or(phone.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn esperanto_sample_loads_expected_phonemes() {
        let eo = variety();
        assert!(
            eo.phonemes
                .phonemes
                .contains_key(&PhonemeId("eo.phoneme.a".into()))
        );
        assert!(
            eo.phonemes
                .phonemes
                .contains_key(&PhonemeId("eo.phoneme.k".into()))
        );
        assert!(
            eo.phonemes
                .phonemes
                .get(&PhonemeId("eo.phoneme.a".into()))
                .expect("a phoneme")
                .aliases
                .iter()
                .any(|alias| alias.system == "esperanto" && alias.symbol == "A")
        );
    }
}
