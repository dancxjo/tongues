use std::collections::HashMap;

mod catalog;
pub mod morphology;

use crate::acoustics::{
    AcousticCueDef, AcousticLandmark, AcousticLandmarkKind, AcousticMeasurement, AcousticProfile,
    AcousticRangeTarget, AcousticTargetModel, AcousticTemporalModel, CueDependency,
    CueDiagnosticity, CueTarget, LandmarkAnchor, LandmarkOrderStep, NumericRange,
    RelativeTimeWindow, SegmentSamplingStrategy, SubsegmentProportion, SubsegmentRole, WeightedCue,
};
use crate::data::lexicons::cmudict::CmuPhoneme;
use crate::data::notation::arpabet::{self, ARPABET};
use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{AcousticCueId, FeatureId, LanguageId, PhoneId, VarietyId};
use crate::orthography::Orthography;
use crate::phonetics::PhoneInventory;
use crate::phonology::{PhonemeAllophone, PhonemeInventory};
use crate::prosody::{ProsodicContext, Stress};
use crate::rules::{
    AllophoneRule, EpenthesisRule, PhonePattern, PhonemePattern, PhonotacticConstraint,
    Phonotactics, RuleCondition, RuleStatus, SyllableShape,
};
use crate::segment::{Environment, SegmentMatcher, SyllablePosition, WordPosition};
use crate::spec::Spec;
use crate::variety::{
    LinguisticVariety, OrthographicUnitKind, OrthographicUnitPronunciation,
    VarietyImplementationStatus, VarietyStatus, WeakFormFollowingContext, WeakFormRule,
    WeakFormStyleContext,
};

const P: PhoneId = PhoneId::borrowed("ipa.phone.p");
const ASPIRATED_P: PhoneId = PhoneId::borrowed("ipa.phone.pʰ");
const UNASPIRATED_P: PhoneId = PhoneId::borrowed("ipa.phone.p˭");
const B: PhoneId = PhoneId::borrowed("ipa.phone.b");
const T: PhoneId = PhoneId::borrowed("ipa.phone.t");
const ASPIRATED_T: PhoneId = PhoneId::borrowed("ipa.phone.tʰ");
const UNASPIRATED_T: PhoneId = PhoneId::borrowed("ipa.phone.t˭");
const D: PhoneId = PhoneId::borrowed("ipa.phone.d");
const K: PhoneId = PhoneId::borrowed("ipa.phone.k");
const ASPIRATED_K: PhoneId = PhoneId::borrowed("ipa.phone.kʰ");
const UNASPIRATED_K: PhoneId = PhoneId::borrowed("ipa.phone.k˭");
const G: PhoneId = PhoneId::borrowed("ipa.phone.ɡ");
const F: PhoneId = PhoneId::borrowed("ipa.phone.f");
const V: PhoneId = PhoneId::borrowed("ipa.phone.v");
const TH: PhoneId = PhoneId::borrowed("ipa.phone.θ");
const SH: PhoneId = PhoneId::borrowed("ipa.phone.ʃ");
const S: PhoneId = PhoneId::borrowed("ipa.phone.s");
const Z: PhoneId = PhoneId::borrowed("ipa.phone.z");
const M: PhoneId = PhoneId::borrowed("ipa.phone.m");
const N: PhoneId = PhoneId::borrowed("ipa.phone.n");
const NG: PhoneId = PhoneId::borrowed("ipa.phone.ŋ");
const L: PhoneId = PhoneId::borrowed("ipa.phone.l");
const DARK_L: PhoneId = PhoneId::borrowed("ipa.phone.ɫ");
const R: PhoneId = PhoneId::borrowed("ipa.phone.ɹ");
const W: PhoneId = PhoneId::borrowed("ipa.phone.w");
const Y: PhoneId = PhoneId::borrowed("ipa.phone.j");
const CH: PhoneId = PhoneId::borrowed("ipa.phone.tʃ");
const JH: PhoneId = PhoneId::borrowed("ipa.phone.dʒ");
const TAP: PhoneId = PhoneId::borrowed("ipa.phone.ɾ");
const SCHWA: PhoneId = PhoneId::borrowed("ipa.phone.ə");
const STRUT: PhoneId = PhoneId::borrowed("ipa.phone.ʌ");
const R_COLORED_SCHWA: PhoneId = PhoneId::borrowed("ipa.phone.ɚ");
const STRESSED_RHOTIC_VOWEL: PhoneId = PhoneId::borrowed("ipa.phone.ɝ");
const SYLLABLE_BREAK: PhoneId = PhoneId::borrowed("ipa.phone.|");
const WORD_BOUNDARY: PhoneId = PhoneId::borrowed("boundary.word");
const LETTER_BOUNDARY: PhoneId = PhoneId::borrowed("boundary.letter");
const PHRASE_PAUSE: PhoneId = PhoneId::borrowed("boundary.phrase_pause");
const TERMINAL_PAUSE: PhoneId = PhoneId::borrowed("boundary.terminal_pause");

const LEGAL_ONSETS: &[&[PhoneId]] = &[
    &[P, L],
    &[B, L],
    &[K, L],
    &[G, L],
    &[F, L],
    &[P, R],
    &[B, R],
    &[T, R],
    &[D, R],
    &[K, R],
    &[G, R],
    &[F, R],
    &[TH, R],
    &[SH, R],
    &[S, P],
    &[S, T],
    &[S, K],
    &[S, UNASPIRATED_P],
    &[S, UNASPIRATED_T],
    &[S, UNASPIRATED_K],
    &[S, L],
    &[S, M],
    &[S, N],
    &[S, W],
    &[S, F],
    &[T, W],
    &[K, W],
    &[G, W],
    &[D, W],
    &[SH, W],
    &[TH, W],
    &[S, P, L],
    &[S, P, R],
    &[S, T, R],
    &[S, K, R],
    &[S, K, W],
    &[S, T, W],
    &[S, UNASPIRATED_P, L],
    &[S, UNASPIRATED_P, R],
    &[S, UNASPIRATED_T, R],
    &[S, UNASPIRATED_K, R],
    &[S, UNASPIRATED_K, W],
    &[S, UNASPIRATED_T, W],
];

const SINGING_ONSET_ADDITIONS: &[&[PhoneId]] = &[&[T, L], &[D, L], &[V, R], &[V, L], &[Z, W]];

const LEGAL_CODAS: &[&[PhoneId]] = &[
    &[N, D],
    &[N, T],
    &[N, Z],
    &[NG, K],
    &[NG, Z],
    &[M, P],
    &[M, Z],
    &[L, D],
    &[L, T],
    &[L, K],
    &[L, P],
    &[L, F],
    &[L, M],
    &[L, N],
    &[L, Z],
    &[S, T],
    &[S, K],
    &[S, P],
    &[F, T],
    &[K, T],
    &[K, S],
    &[P, T],
    &[P, S],
    &[T, S],
    &[D, Z],
    &[R, D],
    &[R, T],
    &[R, K],
    &[R, N],
    &[R, M],
    &[R, Z],
    &[R, P],
    &[R, F],
    &[N, CH],
    &[N, JH],
    &[L, CH],
    &[R, CH],
    &[N, D, Z],
    &[N, T, S],
    &[NG, K, S],
    &[L, D, Z],
    &[L, T, S],
    &[L, K, S],
    &[M, P, T],
    &[M, P, S],
    &[S, T, S],
    &[K, T, S],
    &[NG, TH, S],
    &[NG, K, TH, S],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClusterScope {
    Onset,
    SingingOnset,
    Coda,
}

pub fn variety(id: &str) -> LinguisticVariety {
    let row = catalog::get(id);
    let phonemes = phoneme_inventory(row.id);
    let phones = phone_inventory();
    let acoustic_profile = acoustic_profile(&phonemes, &phones);

    LinguisticVariety {
        id: VarietyId(row.id.into()),
        language: LanguageId("en".into()),
        name: row.name.into(),
        feature_system: FeatureSystem::default(),
        phonemes,
        phones,
        allophone_rules: allophone_rules(row.id),
        epenthesis_rules: epenthesis_rules(),
        weak_forms: weak_forms(row.id),
        orthographic_unit_pronunciations: orthographic_unit_pronunciations(row.id),
        phonotactics: Some(phonotactics(row.singing)),
        orthography: Some(Orthography {
            name: "English Latin orthography".into(),
            ..Default::default()
        }),
        morphology: Some(morphology::english_morphology(row.id)),
        acoustic_profile: Some(acoustic_profile),
        prosody_profile: None,
        status: VarietyStatus::Attested,
        implementation_status: match row.implementation_status {
            catalog::ImplementationStatusSpec::Complete => VarietyImplementationStatus::Complete,
            catalog::ImplementationStatusSpec::StubDerivedFrom(source) => {
                VarietyImplementationStatus::StubDerivedFrom(VarietyId(source.into()))
            }
            catalog::ImplementationStatusSpec::PermissiveProfile => {
                VarietyImplementationStatus::PermissiveProfile
            }
        },
    }
}

fn orthographic_unit_pronunciations(variety_id: &str) -> Vec<OrthographicUnitPronunciation> {
    let letters: &[(char, &[&str])] = &[
        ('A', &["EY1"]),
        ('B', &["B", "IY1"]),
        ('C', &["S", "IY1"]),
        ('D', &["D", "IY1"]),
        ('E', &["IY1"]),
        ('F', &["EH1", "F"]),
        ('G', &["JH", "IY1"]),
        ('H', &["EY1", "CH"]),
        ('I', &["AY1"]),
        ('J', &["JH", "EY1"]),
        ('K', &["K", "EY1"]),
        ('L', &["EH1", "L"]),
        ('M', &["EH1", "M"]),
        ('N', &["EH1", "N"]),
        ('O', &["OW1"]),
        ('P', &["P", "IY1"]),
        ('Q', &["K", "Y", "UW1"]),
        ('R', &["AA1", "R"]),
        ('S', &["EH1", "S"]),
        ('T', &["T", "IY1"]),
        ('U', &["Y", "UW1"]),
        ('V', &["V", "IY1"]),
        ('W', &["D", "AH1", "B", "AH0", "L", "Y", "UW0"]),
        ('X', &["EH1", "K", "S"]),
        ('Y', &["W", "AY1"]),
        ('Z', &["Z", "IY1"]),
    ];
    let digits: &[(char, &[&str])] = &[
        ('0', &["Z", "IH1", "R", "OW0"]),
        ('1', &["W", "AH1", "N"]),
        ('2', &["T", "UW1"]),
        ('3', &["TH", "R", "IY1"]),
        ('4', &["F", "AO1", "R"]),
        ('5', &["F", "AY1", "V"]),
        ('6', &["S", "IH1", "K", "S"]),
        ('7', &["S", "EH1", "V", "AH0", "N"]),
        ('8', &["EY1", "T"]),
        ('9', &["N", "AY1", "N"]),
    ];

    letters
        .iter()
        .map(|(letter, symbols)| {
            orthographic_unit(
                variety_id,
                OrthographicUnitKind::LetterName,
                *letter,
                symbols,
            )
        })
        .chain(digits.iter().map(|(digit, symbols)| {
            orthographic_unit(variety_id, OrthographicUnitKind::DigitName, *digit, symbols)
        }))
        .collect()
}

fn orthographic_unit(
    variety_id: &str,
    kind: OrthographicUnitKind,
    unit: char,
    symbols: &[&str],
) -> OrthographicUnitPronunciation {
    OrthographicUnitPronunciation {
        kind,
        unit: unit.to_string(),
        pronunciation: symbols
            .iter()
            .map(|symbol| arpabet::phoneme_id(variety_id, symbol))
            .collect(),
        cmudict_pronunciation: symbols
            .iter()
            .map(|symbol| CmuPhoneme::parse(symbol))
            .collect(),
    }
}

fn weak_forms(variety_id: &str) -> Vec<WeakFormRule> {
    [
        weak_form(
            "english_weak_the_before_vowel",
            "the",
            &["DH", "IY0"],
            WeakFormFollowingContext::BeforeVowelish,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_the_before_consonant",
            "the",
            &["DH", "AH0"],
            WeakFormFollowingContext::BeforeConsonantish,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_and",
            "and",
            &["AH0", "N", "D"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_a",
            "a",
            &["AH0"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_an",
            "an",
            &["AH0", "N"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_of",
            "of",
            &["AH0", "V"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variety_id,
        ),
        weak_form(
            "english_weak_to_before_consonant",
            "to",
            &["T", "AH0"],
            WeakFormFollowingContext::BeforeConsonantish,
            WeakFormStyleContext::CasualOnly,
            variety_id,
        ),
    ]
    .into()
}

fn weak_form(
    id: &str,
    lexical_item: &str,
    symbols: &[&str],
    following: WeakFormFollowingContext,
    style: WeakFormStyleContext,
    variety_id: &str,
) -> WeakFormRule {
    WeakFormRule {
        id: id.into(),
        lexical_item: lexical_item.into(),
        pronunciation: symbols
            .iter()
            .map(|symbol| arpabet::phoneme_id(variety_id, symbol))
            .collect(),
        cmudict_pronunciation: symbols
            .iter()
            .map(|symbol| CmuPhoneme::parse(symbol))
            .collect(),
        following,
        style,
    }
}

fn phoneme_inventory(variety_id: &str) -> PhonemeInventory {
    let mut phonemes = ARPABET
        .iter()
        .map(|entry| {
            let mut phoneme = arpabet::phoneme_for_entry(variety_id, entry);
            enrich_english_inventory_features(&mut phoneme.features, entry);
            (phoneme.id.clone(), phoneme)
        })
        .collect::<HashMap<_, _>>();
    if let Some(ah) = phonemes.get_mut(&arpabet::phoneme_id(variety_id, "AH")) {
        ah.default_phone = Some(SCHWA);
        if !ah.possible_phones.contains(&SCHWA) {
            ah.possible_phones.insert(0, SCHWA);
        }
    }
    if let Some(er) = phonemes.get_mut(&arpabet::phoneme_id(variety_id, "ER")) {
        er.default_phone = Some(R_COLORED_SCHWA);
        if !er.possible_phones.contains(&R_COLORED_SCHWA) {
            er.possible_phones.insert(0, R_COLORED_SCHWA);
        }
    }
    for rule in allophone_rules(variety_id) {
        let Spec::Known(phoneme_id) = &rule.input.phoneme else {
            continue;
        };
        let Spec::Known(phone_id) = &rule.output.phone else {
            continue;
        };
        if let Some(phoneme) = phonemes.get_mut(phoneme_id) {
            if !phoneme.possible_phones.contains(phone_id) {
                phoneme.possible_phones.push(phone_id.clone());
            }
            phoneme.allophones.push(PhonemeAllophone {
                phone: phone_id.clone(),
                environment: rule.environment.clone(),
                conditions: rule.conditions.clone(),
                confidence: rule.confidence,
                status: rule.status.clone(),
                source_rule_id: Some(rule.id.clone()),
            });
        }
    }
    PhonemeInventory { phonemes }
}

fn phone_inventory() -> PhoneInventory {
    let mut phones = HashMap::new();
    for entry in ARPABET {
        let mut phone = arpabet::phone_for_entry(entry);
        enrich_english_inventory_features(&mut phone.features, entry);
        if matches!(entry.symbol, "AH" | "ER") {
            phone.status = crate::segment::SegmentStatus::Allophonic;
        }
        phones.insert(phone.id.clone(), phone);
    }
    for (phone_ref, base, ipa) in [(SCHWA, "AH", "ə"), (R_COLORED_SCHWA, "ER", "ɚ")] {
        let mut features = arpabet::entry(base)
            .map(arpabet::feature_bundle)
            .unwrap_or_default();
        if let Some(entry) = arpabet::entry(base) {
            enrich_english_inventory_features(&mut features, entry);
        }
        features.values.insert(
            FeatureId("phonology.reduced_vowel".into()),
            Spec::Known(FeatureValue::Bool(true)),
        );
        let status = if phone_ref == SCHWA || phone_ref == R_COLORED_SCHWA {
            crate::segment::SegmentStatus::Core
        } else {
            crate::segment::SegmentStatus::Allophonic
        };
        let phone = crate::phonetics::Phone {
            id: phone_ref,
            ipa: ipa.into(),
            features,
            aliases: Vec::new(),
            status,
        };
        phones.insert(phone.id.clone(), phone);
    }
    for (phone_ref, base, ipa, aspiration) in [
        (ASPIRATED_P, "P", "pʰ", "aspirated"),
        (ASPIRATED_T, "T", "tʰ", "aspirated"),
        (ASPIRATED_K, "K", "kʰ", "aspirated"),
        (UNASPIRATED_P, "P", "p˭", "unaspirated"),
        (UNASPIRATED_T, "T", "t˭", "unaspirated"),
        (UNASPIRATED_K, "K", "k˭", "unaspirated"),
    ] {
        let mut features = allophonic_features_from_arpabet(base);
        put_phonology_category(&mut features, "aspiration", aspiration);
        phones.insert(
            phone_ref.clone(),
            crate::phonetics::Phone {
                id: phone_ref,
                ipa: ipa.into(),
                features,
                aliases: Vec::new(),
                status: crate::segment::SegmentStatus::Allophonic,
            },
        );
    }
    {
        let mut features = allophonic_features_from_arpabet("L");
        put_phonology_category(&mut features, "l_quality", "dark");
        put_phonology_bool(&mut features, "velarized_lateral", true);
        put_phonology_category(
            &mut features,
            "approximant_trajectory",
            "velarized_lateral_approximant",
        );
        phones.insert(
            DARK_L,
            crate::phonetics::Phone {
                id: DARK_L,
                ipa: "ɫ".into(),
                features,
                aliases: Vec::new(),
                status: crate::segment::SegmentStatus::Allophonic,
            },
        );
    }
    for phone_ref in [TAP, SYLLABLE_BREAK] {
        let ipa = phone_symbol(&phone_ref).into();
        let features = if phone_ref == TAP {
            tap_feature_bundle()
        } else if phone_ref == SYLLABLE_BREAK {
            syllable_break_feature_bundle()
        } else {
            Default::default()
        };
        let phone = crate::phonetics::Phone {
            id: phone_ref,
            ipa,
            features,
            aliases: Vec::new(),
            status: crate::segment::SegmentStatus::Allophonic,
        };
        phones.insert(phone.id.clone(), phone);
    }
    for (phone_ref, symbol, boundary_kind, gap_class, silent) in [
        (WORD_BOUNDARY, "|", "word", "none", false),
        (LETTER_BOUNDARY, "|", "letter", "none", false),
        (PHRASE_PAUSE, "‖", "phrase", "phrase_pause", true),
        (TERMINAL_PAUSE, "||", "terminal", "terminal_pause", true),
    ] {
        let phone = crate::phonetics::Phone {
            id: phone_ref,
            ipa: symbol.into(),
            features: boundary_feature_bundle(boundary_kind, gap_class, silent),
            aliases: Vec::new(),
            status: crate::segment::SegmentStatus::Allophonic,
        };
        phones.insert(phone.id.clone(), phone);
    }
    PhoneInventory { phones }
}

fn allophonic_features_from_arpabet(base: &str) -> FeatureBundle {
    let mut features = arpabet::entry(base)
        .map(arpabet::feature_bundle)
        .unwrap_or_default();
    if let Some(entry) = arpabet::entry(base) {
        enrich_english_inventory_features(&mut features, entry);
    }
    features
}

fn enrich_english_inventory_features(features: &mut FeatureBundle, entry: &arpabet::ArpabetEntry) {
    if entry.syllabic {
        let trajectory = formant_trajectory_for_alias(entry.symbol);
        put_phonology_category(features, "formant_trajectory", trajectory);
        put_phonology_bool(features, "diphthong", trajectory != "stable");
        put_phonology_bool(features, "rhoticity", entry.vowel_height == Some("rhotic"));
    } else {
        if matches!(entry.manner, Some("fricative" | "affricate")) {
            put_phonology_category(
                features,
                "frication_spectral_shape",
                frication_spectral_shape_for_entry(entry),
            );
        }
        if matches!(entry.manner, Some("liquid" | "glide")) {
            put_phonology_category(
                features,
                "approximant_trajectory",
                approximant_trajectory_for_entry(entry),
            );
            put_phonology_bool(features, "rhoticity", entry.symbol == "R");
            put_phonology_bool(features, "lateral_resonance", entry.symbol == "L");
        }
    }
}

fn tap_feature_bundle() -> FeatureBundle {
    let mut features = FeatureBundle::default();
    put_phonology_category(&mut features, "major", "consonant");
    put_phonology_bool(&mut features, "syllabic", false);
    put_phonology_category(&mut features, "place", "alveolar");
    put_phonology_category(&mut features, "manner", "tap");
    put_phonology_category(&mut features, "voicing", "voiced");
    features
}

fn syllable_break_feature_bundle() -> FeatureBundle {
    let mut features = FeatureBundle::default();
    put_phonology_category(&mut features, "major", "boundary");
    put_phonology_category(&mut features, "boundary_kind", "syllable");
    put_phonology_category(&mut features, "boundary_gap_class", "none");
    put_phonology_bool(&mut features, "silent_boundary", false);
    put_phonology_bool(&mut features, "syllabic", false);
    features
}

fn boundary_feature_bundle(boundary_kind: &str, gap_class: &str, silent: bool) -> FeatureBundle {
    let mut features = FeatureBundle::default();
    put_phonology_category(&mut features, "major", "boundary");
    put_phonology_category(&mut features, "boundary_kind", boundary_kind);
    put_phonology_category(&mut features, "boundary_gap_class", gap_class);
    put_phonology_bool(&mut features, "silent_boundary", silent);
    put_phonology_bool(&mut features, "syllabic", false);
    features
}

fn put_phonology_category(features: &mut FeatureBundle, name: &str, value: &str) {
    features.values.insert(
        FeatureId(format!("phonology.{name}")),
        Spec::Known(FeatureValue::Category(value.into())),
    );
}

fn put_phonology_bool(features: &mut FeatureBundle, name: &str, value: bool) {
    features.values.insert(
        FeatureId(format!("phonology.{name}")),
        Spec::Known(FeatureValue::Bool(value)),
    );
}

fn acoustic_profile(phonemes: &PhonemeInventory, phones: &PhoneInventory) -> AcousticProfile {
    let mut cues = HashMap::new();
    for cue in acoustic_cues() {
        cues.insert(cue.id.clone(), cue);
    }

    let mut phone_models = HashMap::new();
    let mut phoneme_models = HashMap::new();
    for phoneme in phonemes.phonemes.values() {
        if let Some(model) = acoustic_model_from_features(&phoneme.features, &phoneme.notation) {
            phoneme_models.insert(phoneme.id.clone(), model);
        }
        for phone_id in phoneme
            .default_phone
            .iter()
            .chain(phoneme.possible_phones.iter())
        {
            if phone_models.contains_key(phone_id) {
                continue;
            }
            let model = phones
                .phones
                .get(phone_id)
                .and_then(|phone| {
                    acoustic_model_from_features(&phone.features, &format!("[{}]", phone.ipa))
                })
                .or_else(|| acoustic_model_from_features(&phoneme.features, &phoneme.notation));
            if let Some(model) = model {
                phone_models.insert(phone_id.clone(), model);
            }
        }
    }
    for (phone_id, phone) in &phones.phones {
        if phone_models.contains_key(phone_id) {
            continue;
        }
        if let Some(model) =
            acoustic_model_from_features(&phone.features, &format!("[{}]", phone.ipa))
        {
            phone_models.insert(phone_id.clone(), model);
        }
    }

    AcousticProfile {
        cues,
        phone_models,
        phoneme_models,
    }
}

fn acoustic_cues() -> Vec<AcousticCueDef> {
    vec![
        cue(
            "acoustic.cue.f1_region",
            "first formant region",
            "acoustic.f1_region",
            vec![CueTarget::Feature(FeatureId(
                "phonology.vowel_height".into(),
            ))],
            Some("Low F1 is a common correlate of high vowels.".into()),
        ),
        cue(
            "acoustic.cue.f2_region",
            "second formant region",
            "acoustic.f2_region",
            vec![CueTarget::Feature(FeatureId(
                "phonology.vowel_backness".into(),
            ))],
            Some("High F2 tends to mark front vowels; low F2 tends to mark back/rounded vowels.".into()),
        ),
        cue(
            "acoustic.cue.rounding_resonance",
            "rounding resonance",
            "acoustic.rounding_resonance",
            vec![CueTarget::Feature(FeatureId("phonology.roundedness".into()))],
            Some("Lip rounding usually lowers upper formant energy and reinforces the [u] vs [i] split.".into()),
        ),
        cue(
            "acoustic.cue.periodic_voicing",
            "periodic voicing",
            "acoustic.periodic_voicing",
            vec![
                CueTarget::Feature(FeatureId("phonology.voicing".into())),
                CueTarget::Feature(FeatureId("phonology.syllabic".into())),
            ],
            Some("Regular low-frequency periodicity is a strong cue for voiced sonorants and vowels.".into()),
        ),
        cue(
            "acoustic.cue.sonority_peak",
            "sonority peak",
            "acoustic.sonority_peak",
            vec![
                CueTarget::Feature(FeatureId("phonology.syllabic".into())),
                CueTarget::Stress,
            ],
            Some("Syllable nuclei tend to carry local sonority, energy, and periodicity maxima.".into()),
        ),
        cue(
            "acoustic.cue.vowel_nucleus",
            "vowel nucleus",
            "acoustic.vowel_nucleus",
            vec![CueTarget::Feature(FeatureId("phonology.syllabic".into()))],
            Some("A vowel nucleus is an alignment anchor: stable voicing plus vowel-like formants near the syllable peak.".into()),
        ),
        cue(
            "acoustic.cue.formant_trajectory",
            "formant trajectory",
            "acoustic.formant_trajectory",
            vec![CueTarget::Feature(FeatureId("phonology.diphthong".into()))],
            Some("Diphthongs are better matched by formant movement than by a single steady vowel target.".into()),
        ),
        cue(
            "acoustic.cue.f3_region",
            "third formant region",
            "acoustic.f3_region",
            vec![CueTarget::Feature(FeatureId("phonology.rhoticity".into()))],
            Some("A lowered F3 is a useful cue for English r-colored vowels.".into()),
        ),
        cue(
            "acoustic.cue.vowel_reduction",
            "vowel reduction",
            "acoustic.vowel_reduction",
            vec![CueTarget::Feature(FeatureId("phonology.reduced_vowel".into()))],
            Some("Reduced vowels tend toward central formants and can have weaker sonority peaks.".into()),
        ),
        cue(
            "acoustic.cue.consonant_place_transition",
            "consonant place transition",
            "acoustic.consonant_place",
            vec![CueTarget::Feature(FeatureId("phonology.place".into()))],
            Some("Neighboring vowel transitions help locate place of articulation for consonants.".into()),
        ),
        cue(
            "acoustic.cue.stop_burst_spectral_shape",
            "stop burst spectral shape",
            "acoustic.stop_burst_spectral_shape",
            vec![
                CueTarget::Feature(FeatureId("phonology.place".into())),
                CueTarget::Feature(FeatureId("phonology.manner".into())),
            ],
            Some("The spectral balance of a stop burst carries useful place information.".into()),
        ),
        cue(
            "acoustic.cue.place_formant_locus",
            "place formant locus",
            "acoustic.place_formant_locus",
            vec![CueTarget::Feature(FeatureId("phonology.place".into()))],
            Some("Transitions into and out of neighboring vowels provide a coarse place target.".into()),
        ),
        cue(
            "acoustic.cue.frication_noise",
            "frication noise",
            "acoustic.frication_noise",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Sustained aperiodic noise is a core cue for fricatives and the fricative portion of affricates.".into()),
        ),
        cue(
            "acoustic.cue.frication_spectral_shape",
            "frication spectral shape",
            "acoustic.frication_spectral_shape",
            vec![
                CueTarget::Feature(FeatureId("phonology.place".into())),
                CueTarget::Feature(FeatureId("phonology.manner".into())),
            ],
            Some("Sibilants, labiodentals, dentals, and glottals differ in the spectral shape of their noise.".into()),
        ),
        cue(
            "acoustic.cue.frication_spectral_skew",
            "frication spectral skew",
            "acoustic.frication_spectral_skew",
            vec![
                CueTarget::Feature(FeatureId("phonology.place".into())),
                CueTarget::Feature(FeatureId("phonology.manner".into())),
            ],
            Some("Spectral skew helps separate high anterior sibilants from postalveolar and diffuse fricatives.".into()),
        ),
        cue(
            "acoustic.cue.affricate_release",
            "affricate release",
            "acoustic.affricate_release",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Affricates combine a stop-like closure and release with following frication.".into()),
        ),
        cue(
            "acoustic.cue.affricate_closure_to_frication_timing",
            "affricate closure-to-frication timing",
            "acoustic.affricate_closure_to_frication_timing",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Affricates are aligned by a short interval from stop release into sustained frication.".into()),
        ),
        cue(
            "acoustic.cue.nasal_murmur",
            "nasal murmur",
            "acoustic.nasal_murmur",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Nasals have low-frequency voicing energy shaped by nasal resonances.".into()),
        ),
        cue(
            "acoustic.cue.nasal_antiresonance",
            "nasal antiresonance",
            "acoustic.nasal_antiresonance",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Nasal coupling introduces spectral zeros that help distinguish nasal consonants from oral sonorants.".into()),
        ),
        cue(
            "acoustic.cue.nasal_place",
            "nasal place",
            "acoustic.nasal_place",
            vec![CueTarget::Feature(FeatureId("phonology.place".into()))],
            Some("Nasal place is weak but can be inferred from murmur spectrum and adjacent vowel transitions.".into()),
        ),
        cue(
            "acoustic.cue.nasal_place_transition",
            "nasal place transition",
            "acoustic.nasal_place_transition",
            vec![CueTarget::Feature(FeatureId("phonology.place".into()))],
            Some("Vowel transitions adjacent to nasal closures provide a place cue alongside murmur and antiresonance.".into()),
        ),
        cue(
            "acoustic.cue.approximant_formants",
            "approximant formants",
            "acoustic.approximant_formants",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Liquids and glides are tracked by smooth voiced formant structure and transitions.".into()),
        ),
        cue(
            "acoustic.cue.approximant_formant_transition_detail",
            "approximant formant transition detail",
            "acoustic.approximant_formant_transition_detail",
            vec![
                CueTarget::Feature(FeatureId("phonology.manner".into())),
                CueTarget::Feature(FeatureId("phonology.place".into())),
            ],
            Some("English glides and liquids have different F2/F3 transition signatures.".into()),
        ),
        cue(
            "acoustic.cue.tap_closure",
            "tap closure",
            "acoustic.tap_closure",
            vec![CueTarget::Phone(TAP)],
            Some("A tap is expected to have a very brief closure rather than a full stop closure interval.".into()),
        ),
        cue(
            "acoustic.cue.segment_boundary",
            "segment boundary",
            "acoustic.segment_boundary",
            vec![CueTarget::Boundary],
            Some("Boundary phones align to timing discontinuities rather than speech energy targets.".into()),
        ),
        cue(
            "acoustic.cue.boundary_gap",
            "boundary gap",
            "acoustic.boundary_gap",
            vec![CueTarget::Boundary],
            Some("A boundary may coincide with a gap, discontinuity, or only a symbolic alignment point.".into()),
        ),
        cue(
            "acoustic.cue.stop_closure",
            "stop closure",
            "acoustic.stop_closure",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("A low-energy closure interval is a core stop landmark.".into()),
        ),
        cue(
            "acoustic.cue.release_burst",
            "release burst",
            "acoustic.release_burst",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Transient burst energy near release helps distinguish stops from continuants.".into()),
        ),
        cue(
            "acoustic.cue.voice_onset_time",
            "voice onset time",
            "acoustic.vot_class",
            vec![
                CueTarget::Feature(FeatureId("phonology.manner".into())),
                CueTarget::Feature(FeatureId("phonology.voicing".into())),
            ],
            Some("VOT separates many English voiced and voiceless stops, but varies with context.".into()),
        ),
        cue(
            "acoustic.cue.closure_voicing",
            "closure voicing",
            "acoustic.voicing_during_closure",
            vec![CueTarget::Feature(FeatureId("phonology.voicing".into()))],
            Some("Periodic low-frequency energy during closure is evidence for a voiced stop.".into()),
        ),
        cue(
            "acoustic.cue.aspiration_noise",
            "aspiration noise",
            "acoustic.aspiration_present",
            vec![
                CueTarget::Feature(FeatureId("phonology.manner".into())),
                CueTarget::Feature(FeatureId("phonology.voicing".into())),
            ],
            Some("Post-release aperiodic breath noise is expected for many English voiceless stops in stressed onsets, but not everywhere.".into()),
        ),
    ]
}

fn acoustic_model_from_features(
    features: &FeatureBundle,
    label: &str,
) -> Option<AcousticTargetModel> {
    match phonology_category(features, "major") {
        Some("vowel") => Some(vowel_model(features, label)),
        Some("consonant") => Some(consonant_model(features, label)),
        Some("boundary") => Some(boundary_model(features, label)),
        _ => None,
    }
}

fn vowel_model(features: &FeatureBundle, label: &str) -> AcousticTargetModel {
    let height = phonology_category(features, "vowel_height");
    let backness = phonology_category(features, "vowel_backness");
    let roundedness = phonology_category(features, "roundedness");
    let trajectory = phonology_category(features, "formant_trajectory").unwrap_or("stable");
    let rhotic = phonology_bool(features, "rhoticity").unwrap_or_else(|| height == Some("rhotic"));
    let reduced = phonology_bool(features, "reduced_vowel").unwrap_or(false);
    let mut expected_features = acoustic_feature_bundle(&[
        (
            "f1_region",
            Spec::Known(FeatureValue::Category(f1_region(height).into())),
        ),
        (
            "f2_region",
            Spec::Known(FeatureValue::Category(f2_region(backness).into())),
        ),
        ("rounding_resonance", rounding_resonance(roundedness)),
        ("periodic_voicing", Spec::Known(FeatureValue::Bool(true))),
        (
            "sonority_peak",
            if reduced {
                Spec::Variable(vec![FeatureValue::Bool(false), FeatureValue::Bool(true)])
            } else {
                Spec::Known(FeatureValue::Bool(true))
            },
        ),
        ("vowel_nucleus", Spec::Known(FeatureValue::Bool(true))),
        (
            "formant_trajectory",
            Spec::Known(FeatureValue::Category(trajectory.into())),
        ),
        ("rhoticity", Spec::Known(FeatureValue::Bool(rhotic))),
    ]);
    if reduced {
        put_acoustic_feature(
            &mut expected_features,
            "vowel_reduction",
            Spec::Known(FeatureValue::Bool(true)),
        );
    }
    if rhotic {
        put_acoustic_feature(
            &mut expected_features,
            "f3_region",
            Spec::Known(FeatureValue::Category("low".into())),
        );
    }

    let mut weighted_cues = weighted_cues(&[
        ("acoustic.cue.f1_region", 0.8),
        ("acoustic.cue.f2_region", 1.0),
        ("acoustic.cue.rounding_resonance", 0.5),
        ("acoustic.cue.periodic_voicing", 0.8),
        ("acoustic.cue.sonority_peak", 0.9),
        ("acoustic.cue.vowel_nucleus", 1.0),
    ]);
    if reduced {
        weighted_cues.push(weighted_cue("acoustic.cue.vowel_reduction", 0.8));
    }
    if trajectory != "stable" {
        weighted_cues.push(weighted_cue("acoustic.cue.formant_trajectory", 0.9));
    }
    if rhotic {
        weighted_cues.push(weighted_cue("acoustic.cue.f3_region", 0.9));
    }

    let range_targets = vowel_formant_targets(label);
    let mut landmarks = vec![
        vowel_target_landmark(&range_targets),
        syllable_nucleus_landmark(),
    ];
    if trajectory != "stable" {
        landmarks.push(formant_trajectory_landmark(trajectory));
    }
    if rhotic {
        landmarks.push(rhotic_target_landmark());
    }

    AcousticTargetModel {
        expected_features,
        weighted_cues,
        landmarks,
        range_targets,
        temporal: vowel_temporal_model(trajectory),
        notes: Some(format!(
            "Vowel nucleus evidence for {label}: {:?} height, {:?} backness, {:?} rounding, {trajectory} trajectory.",
            height, backness, roundedness
        )),
    }
}

fn f1_region(height: Option<&str>) -> &'static str {
    match height {
        Some("high") => "low",
        Some("mid") | Some("rhotic") => "mid",
        Some("low") => "high",
        _ => "mid",
    }
}

fn f2_region(backness: Option<&str>) -> &'static str {
    match backness {
        Some("front") => "high",
        Some("central") => "mid",
        Some("back") => "low",
        _ => "mid",
    }
}

fn rounding_resonance(roundedness: Option<&str>) -> Spec<FeatureValue> {
    match roundedness {
        Some("rounded") => Spec::Known(FeatureValue::Category("present".into())),
        Some("unrounded") => Spec::Known(FeatureValue::Category("absent".into())),
        _ => Spec::Unspecified,
    }
}

fn vowel_formant_targets(label: &str) -> Vec<AcousticRangeTarget> {
    let phone = label.trim_matches(|ch| matches!(ch, '[' | ']' | '/'));
    match phone {
        "iː" => formant_targets((240.0, 350.0), (2200.0, 3000.0), (2800.0, 3600.0)),
        "ɪ" => formant_targets((350.0, 500.0), (1700.0, 2300.0), (2500.0, 3300.0)),
        "eɪ" => formant_targets((350.0, 550.0), (1800.0, 2600.0), (2500.0, 3400.0)),
        "ɛ" => formant_targets((500.0, 700.0), (1700.0, 2400.0), (2400.0, 3300.0)),
        "æ" => formant_targets((650.0, 900.0), (1700.0, 2500.0), (2400.0, 3300.0)),
        "ʌ" => formant_targets((550.0, 800.0), (1100.0, 1700.0), (2300.0, 3200.0)),
        "ə" => formant_targets((450.0, 650.0), (1200.0, 1800.0), (2200.0, 3100.0)),
        "ɝ" => formant_targets((450.0, 650.0), (1200.0, 1700.0), (1400.0, 2200.0)),
        "ɚ" => formant_targets((400.0, 650.0), (1100.0, 1700.0), (1300.0, 2200.0)),
        "ɑ" => formant_targets((650.0, 900.0), (900.0, 1400.0), (2300.0, 3200.0)),
        "ɔ" => formant_targets((500.0, 750.0), (800.0, 1300.0), (2200.0, 3100.0)),
        "oʊ" => formant_targets((350.0, 600.0), (800.0, 1300.0), (2200.0, 3100.0)),
        "ʊ" => formant_targets((350.0, 550.0), (900.0, 1500.0), (2200.0, 3100.0)),
        "uː" => formant_targets((250.0, 400.0), (600.0, 1200.0), (2100.0, 3100.0)),
        "aʊ" => formant_targets((500.0, 850.0), (900.0, 1700.0), (2200.0, 3300.0)),
        "aɪ" => formant_targets((500.0, 850.0), (1500.0, 2500.0), (2400.0, 3400.0)),
        "ɔɪ" => formant_targets((400.0, 750.0), (1200.0, 2300.0), (2300.0, 3300.0)),
        _ => Vec::new(),
    }
}

fn formant_targets(f1: (f32, f32), f2: (f32, f32), f3: (f32, f32)) -> Vec<AcousticRangeTarget> {
    vec![
        range_target(
            AcousticMeasurement::Formant { index: 1 },
            f1.0,
            f1.1,
            "Hz",
            0.65,
            Some(
                "Broad General American vowel target band; speaker-normalized fitting is expected.",
            ),
        ),
        range_target(
            AcousticMeasurement::Formant { index: 2 },
            f2.0,
            f2.1,
            "Hz",
            0.65,
            Some(
                "Broad General American vowel target band; speaker-normalized fitting is expected.",
            ),
        ),
        range_target(
            AcousticMeasurement::Formant { index: 3 },
            f3.0,
            f3.1,
            "Hz",
            0.6,
            Some("F3 is included as a guide range and is especially important for rhotic targets."),
        ),
    ]
}

fn formant_trajectory_for_alias(symbol: &str) -> &'static str {
    match symbol {
        "AW" => "low_central_to_high_back",
        "AY" => "low_front_to_high_front",
        "EY" => "mid_front_to_high_front",
        "OW" => "mid_back_to_high_back",
        "OY" => "low_back_to_high_front",
        _ => "stable",
    }
}

fn vowel_temporal_model(trajectory: &str) -> AcousticTemporalModel {
    if trajectory == "stable" {
        AcousticTemporalModel {
            landmark_order: vec![landmark_order(AcousticLandmarkKind::VowelTarget, true)],
            sampling_strategy: Some(SegmentSamplingStrategy::UseMidpoint),
            ..Default::default()
        }
    } else {
        AcousticTemporalModel {
            landmark_order: vec![
                landmark_order(AcousticLandmarkKind::FormantTransition, true),
                landmark_order(AcousticLandmarkKind::VowelTarget, true),
            ],
            subsegments: vec![
                subsegment(
                    SubsegmentRole::VowelOnsetTransition,
                    0.25,
                    0.4,
                    Some("Initial vowel target movement for a diphthong."),
                ),
                subsegment(
                    SubsegmentRole::VowelSteadyTarget,
                    0.1,
                    0.3,
                    Some(
                        "Brief midpoint region; do not treat diphthongs as a single steady vowel.",
                    ),
                ),
                subsegment(
                    SubsegmentRole::VowelOffsetTransition,
                    0.35,
                    0.55,
                    Some("Final offglide movement carries much of the diphthong identity."),
                ),
            ],
            sampling_strategy: Some(SegmentSamplingStrategy::UseFullTrajectory),
        }
    }
}

fn consonant_temporal_model(
    manner: &str,
    voicing: &str,
    aspiration: Option<&str>,
    approximant_trajectory: &str,
) -> AcousticTemporalModel {
    match manner {
        "stop" => {
            let aspirated = aspiration == Some("aspirated")
                || (voicing == "voiceless" && aspiration != Some("unaspirated"));
            AcousticTemporalModel {
                landmark_order: vec![
                    landmark_order(AcousticLandmarkKind::Closure, true),
                    landmark_order(AcousticLandmarkKind::ReleaseBurst, true),
                    landmark_order(AcousticLandmarkKind::Aspiration, aspirated),
                    landmark_order(AcousticLandmarkKind::VoicingOnset, true),
                ],
                subsegments: stop_subsegments(aspirated),
                sampling_strategy: Some(SegmentSamplingStrategy::UseOffsetTransition),
            }
        }
        "affricate" => AcousticTemporalModel {
            landmark_order: vec![
                landmark_order(AcousticLandmarkKind::Closure, true),
                landmark_order(AcousticLandmarkKind::ReleaseBurst, true),
                landmark_order(AcousticLandmarkKind::AperiodicNoise, true),
                landmark_order(AcousticLandmarkKind::VoicingOnset, voicing == "voiced"),
            ],
            subsegments: vec![
                subsegment(
                    SubsegmentRole::Closure,
                    0.35,
                    0.55,
                    Some("Stop-like closure portion of an affricate."),
                ),
                subsegment(
                    SubsegmentRole::Burst,
                    0.02,
                    0.08,
                    Some("Short release burst between closure and frication."),
                ),
                subsegment(
                    SubsegmentRole::Frication,
                    0.35,
                    0.6,
                    Some("Sustained fricative portion after release."),
                ),
            ],
            sampling_strategy: Some(SegmentSamplingStrategy::UseFullTrajectory),
        },
        "tap" => AcousticTemporalModel {
            landmark_order: vec![landmark_order(AcousticLandmarkKind::Closure, true)],
            subsegments: vec![subsegment(
                SubsegmentRole::TapClosure,
                0.55,
                0.95,
                Some("Tap is dominated by one very brief closure gesture."),
            )],
            sampling_strategy: Some(SegmentSamplingStrategy::UseMidpoint),
        },
        "nasal" => AcousticTemporalModel {
            landmark_order: vec![landmark_order(AcousticLandmarkKind::PeriodicVoicing, true)],
            sampling_strategy: Some(SegmentSamplingStrategy::UseOnsetAndOffsetTransitions),
            ..Default::default()
        },
        "liquid" | "glide" => AcousticTemporalModel {
            landmark_order: vec![landmark_order(
                AcousticLandmarkKind::FormantTransition,
                true,
            )],
            sampling_strategy: Some(if approximant_trajectory == "smooth_approximant" {
                SegmentSamplingStrategy::UseMidpoint
            } else {
                SegmentSamplingStrategy::UseFullTrajectory
            }),
            ..Default::default()
        },
        "fricative" => AcousticTemporalModel {
            landmark_order: vec![landmark_order(AcousticLandmarkKind::AperiodicNoise, true)],
            sampling_strategy: Some(SegmentSamplingStrategy::UseMidpoint),
            ..Default::default()
        },
        _ => AcousticTemporalModel::default(),
    }
}

fn stop_subsegments(aspirated: bool) -> Vec<SubsegmentProportion> {
    if aspirated {
        vec![
            subsegment(
                SubsegmentRole::Closure,
                0.45,
                0.7,
                Some("Low-energy closure before release."),
            ),
            subsegment(
                SubsegmentRole::Burst,
                0.02,
                0.08,
                Some("Short stop release burst."),
            ),
            subsegment(
                SubsegmentRole::Aspiration,
                0.15,
                0.4,
                Some("Post-release aspiration before stable voicing."),
            ),
            subsegment(
                SubsegmentRole::VoiceOnsetLag,
                0.05,
                0.25,
                Some("Lag from release toward periodic voicing onset."),
            ),
        ]
    } else {
        vec![
            subsegment(
                SubsegmentRole::Closure,
                0.6,
                0.9,
                Some("Low-energy closure dominates unaspirated stop timing."),
            ),
            subsegment(
                SubsegmentRole::Burst,
                0.02,
                0.1,
                Some("Short release burst."),
            ),
            subsegment(
                SubsegmentRole::VoiceOnsetLag,
                0.0,
                0.2,
                Some("Short lag or prevoicing interval."),
            ),
        ]
    }
}

fn landmark_order(kind: AcousticLandmarkKind, required: bool) -> LandmarkOrderStep {
    LandmarkOrderStep { kind, required }
}

fn subsegment(
    role: SubsegmentRole,
    min: f32,
    max: f32,
    notes: Option<&str>,
) -> SubsegmentProportion {
    SubsegmentProportion {
        role,
        proportion: NumericRange {
            min,
            max,
            unit: "proportion".into(),
        },
        notes: notes.map(str::to_owned),
    }
}

fn consonant_model(features: &FeatureBundle, label: &str) -> AcousticTargetModel {
    let manner = phonology_category(features, "manner").unwrap_or("consonant");
    let place = phonology_category(features, "place").unwrap_or("unspecified");
    let voicing = phonology_category(features, "voicing").unwrap_or("unspecified");
    let aspiration = phonology_category(features, "aspiration");
    let frication_spectral_shape = phonology_category(features, "frication_spectral_shape")
        .unwrap_or_else(|| frication_spectral_shape_for_place(place));
    let approximant_trajectory =
        phonology_category(features, "approximant_trajectory").unwrap_or("smooth_approximant");
    let rhotic = phonology_bool(features, "rhoticity").unwrap_or(false);
    let lateral = phonology_bool(features, "lateral_resonance").unwrap_or(false);
    let l_quality = phonology_category(features, "l_quality");
    let partial_devoicing = phonology_bool(features, "partial_devoicing").unwrap_or(false);
    let mut expected_features = acoustic_feature_bundle(&[
        ("consonant", Spec::Known(FeatureValue::Bool(true))),
        (
            "consonant_manner",
            Spec::Known(FeatureValue::Category(manner.into())),
        ),
        (
            "consonant_place",
            Spec::Known(FeatureValue::Category(place.into())),
        ),
        (
            "consonant_voicing",
            Spec::Known(FeatureValue::Category(voicing.into())),
        ),
        (
            "periodic_voicing",
            consonant_periodic_voicing(manner, voicing),
        ),
        (
            "place_formant_locus",
            Spec::Known(FeatureValue::Category(place_formant_locus(place).into())),
        ),
    ]);
    if partial_devoicing {
        put_acoustic_feature(
            &mut expected_features,
            "partial_devoicing",
            Spec::Known(FeatureValue::Bool(true)),
        );
    }
    let mut weighted_cues = weighted_cues(&[
        ("acoustic.cue.consonant_place_transition", 0.5),
        ("acoustic.cue.place_formant_locus", 0.5),
        ("acoustic.cue.periodic_voicing", 0.5),
    ]);
    let mut landmarks = Vec::new();
    let mut range_targets = Vec::new();

    match manner {
        "stop" => add_stop_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            &mut range_targets,
            place,
            voicing,
            aspiration,
        ),
        "fricative" => add_fricative_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            &mut range_targets,
            frication_spectral_shape,
            voicing,
        ),
        "affricate" => add_affricate_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            &mut range_targets,
            place,
            voicing,
            frication_spectral_shape,
        ),
        "nasal" => add_nasal_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            &mut range_targets,
            place,
        ),
        "liquid" | "glide" => add_approximant_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            &mut range_targets,
            manner,
            approximant_trajectory,
            rhotic,
            lateral,
            l_quality,
            label,
        ),
        "tap" => add_tap_acoustics(
            &mut expected_features,
            &mut weighted_cues,
            &mut landmarks,
            &mut range_targets,
        ),
        _ => {}
    }

    AcousticTargetModel {
        expected_features,
        weighted_cues,
        landmarks,
        range_targets,
        temporal: consonant_temporal_model(manner, voicing, aspiration, approximant_trajectory),
        notes: Some(format!(
            "Consonant evidence for {label}: {place} {manner}, {voicing}."
        )),
    }
}

fn consonant_periodic_voicing(manner: &str, voicing: &str) -> Spec<FeatureValue> {
    match (manner, voicing) {
        ("nasal" | "liquid" | "glide", "voiced") => Spec::Known(FeatureValue::Bool(true)),
        ("stop" | "fricative" | "affricate", "voiced") => {
            Spec::Variable(vec![FeatureValue::Bool(true), FeatureValue::Bool(false)])
        }
        (_, "voiceless") => Spec::Known(FeatureValue::Bool(false)),
        _ => Spec::Unspecified,
    }
}

fn boundary_model(features: &FeatureBundle, label: &str) -> AcousticTargetModel {
    let boundary_kind = phonology_category(features, "boundary_kind").unwrap_or("segment");
    let gap_class = phonology_category(features, "boundary_gap_class").unwrap_or("none");
    let silent = phonology_bool(features, "silent_boundary").unwrap_or(false);
    AcousticTargetModel {
        expected_features: acoustic_feature_bundle(&[
            (
                "segment_boundary",
                Spec::Known(FeatureValue::Category(boundary_kind.into())),
            ),
            (
                "boundary_gap",
                Spec::Known(FeatureValue::Category(gap_class.into())),
            ),
            ("silent_boundary", Spec::Known(FeatureValue::Bool(silent))),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.segment_boundary", 1.0),
            ("acoustic.cue.boundary_gap", if silent { 0.9 } else { 0.25 }),
        ]),
        landmarks: vec![boundary_landmark(boundary_kind, gap_class, silent)],
        range_targets: boundary_range_targets(gap_class),
        temporal: AcousticTemporalModel {
            sampling_strategy: Some(SegmentSamplingStrategy::UseMidpoint),
            ..Default::default()
        },
        notes: Some(format!(
            "Boundary evidence for {label}: {boundary_kind} boundary with {gap_class} gap class."
        )),
    }
}

fn boundary_range_targets(gap_class: &str) -> Vec<AcousticRangeTarget> {
    match gap_class {
        "none" => vec![range_target(
            AcousticMeasurement::SilenceDuration,
            0.0,
            20.0,
            "ms",
            0.8,
            Some("Boundary-only alignment point; do not require an audible silent gap."),
        )],
        "brief" => vec![range_target(
            AcousticMeasurement::SilenceDuration,
            20.0,
            120.0,
            "ms",
            0.65,
            Some("Brief boundary gap."),
        )],
        "phrase_pause" => vec![range_target(
            AcousticMeasurement::SilenceDuration,
            120.0,
            450.0,
            "ms",
            0.75,
            Some("Phrase-medial pause such as comma, semicolon, or colon."),
        )],
        "terminal_pause" => vec![range_target(
            AcousticMeasurement::SilenceDuration,
            350.0,
            1200.0,
            "ms",
            0.8,
            Some("Terminal pause after sentence-final punctuation."),
        )],
        _ => Vec::new(),
    }
}

fn place_formant_locus(place: &str) -> &'static str {
    match place {
        "bilabial" | "labiodental" => "labial_low_f2",
        "dental" | "alveolar" => "coronal_fronted",
        "postalveolar" | "palatal" => "postalveolar_palatal",
        "velar" => "velar_pinched",
        "glottal" => "glottal_source",
        _ => "unspecified",
    }
}

fn stop_burst_spectral_shape(place: &str) -> &'static str {
    match place {
        "bilabial" => "diffuse_falling",
        "alveolar" | "dental" => "diffuse_rising",
        "postalveolar" | "palatal" => "mid_high_compact",
        "velar" => "compact",
        "glottal" => "weak_or_absent",
        _ => "unspecified",
    }
}

fn nasal_place_cue(place: &str) -> &'static str {
    match place {
        "bilabial" => "labial_murmur",
        "alveolar" | "dental" => "coronal_murmur",
        "velar" => "velar_murmur",
        _ => "unspecified",
    }
}

fn add_stop_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    range_targets: &mut Vec<AcousticRangeTarget>,
    place: &str,
    voicing: &str,
    aspiration: Option<&str>,
) {
    let voiced = voicing == "voiced";
    let aspiration_present = match (voiced, aspiration) {
        (true, _) => Spec::Known(FeatureValue::Bool(false)),
        (false, Some("aspirated")) => Spec::Known(FeatureValue::Bool(true)),
        (false, Some("unaspirated")) => Spec::Known(FeatureValue::Bool(false)),
        (false, _) => Spec::Variable(vec![FeatureValue::Bool(false), FeatureValue::Bool(true)]),
    };
    let closure_voicing = if voiced {
        Spec::Variable(vec![FeatureValue::Bool(true), FeatureValue::Bool(false)])
    } else {
        Spec::Known(FeatureValue::Bool(false))
    };
    put_acoustic_feature(
        features,
        "stop_closure",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "release_burst",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(features, "voicing_during_closure", closure_voicing.clone());
    put_acoustic_feature(features, "vot_class", stop_vot_class(voiced));
    put_acoustic_feature(
        features,
        "stop_burst_spectral_shape",
        Spec::Known(FeatureValue::Category(
            stop_burst_spectral_shape(place).into(),
        )),
    );
    put_acoustic_feature(features, "aspiration_present", aspiration_present.clone());

    cues.extend(weighted_cues(&[
        ("acoustic.cue.stop_closure", 1.0),
        ("acoustic.cue.release_burst", 0.9),
        ("acoustic.cue.stop_burst_spectral_shape", 0.8),
        ("acoustic.cue.voice_onset_time", 0.9),
        (
            "acoustic.cue.closure_voicing",
            if voiced { 0.8 } else { 0.5 },
        ),
    ]));
    if !voiced && aspiration != Some("unaspirated") {
        cues.push(weighted_cue("acoustic.cue.aspiration_noise", 0.6));
    }

    let closure_duration = stop_closure_duration_target(voiced);
    range_targets.push(stop_vot_target(place, voiced));
    range_targets.push(closure_duration.clone());

    landmarks.push(closure_landmark(closure_voicing, &[closure_duration]));
    landmarks.push(release_burst_landmark(stop_burst_spectral_shape(place)));
    if !voiced && aspiration_present != Spec::Known(FeatureValue::Bool(false)) {
        landmarks.push(aspiration_landmark());
    }
    landmarks.push(voicing_onset_landmark(voiced, aspiration_present));
}

fn stop_vot_class(voiced: bool) -> Spec<FeatureValue> {
    if voiced {
        Spec::Variable(vec![
            FeatureValue::Category("prevoiced".into()),
            FeatureValue::Category("short_lag".into()),
        ])
    } else {
        Spec::Variable(vec![
            FeatureValue::Category("short_lag".into()),
            FeatureValue::Category("long_lag".into()),
        ])
    }
}

fn stop_vot_target(place: &str, voiced: bool) -> AcousticRangeTarget {
    let (min, max) = match (place, voiced) {
        ("bilabial", true) => (-80.0, 20.0),
        ("alveolar" | "dental", true) => (-70.0, 25.0),
        ("velar", true) => (-70.0, 30.0),
        (_, true) => (-70.0, 30.0),
        ("bilabial", false) => (25.0, 85.0),
        ("alveolar" | "dental", false) => (30.0, 100.0),
        ("velar", false) => (40.0, 120.0),
        ("glottal", false) => (0.0, 40.0),
        (_, false) => (25.0, 100.0),
    };
    range_target(
        AcousticMeasurement::VoiceOnsetTime,
        min,
        max,
        "ms",
        0.7,
        Some("English stop VOT range; negative values represent prevoicing."),
    )
}

fn stop_closure_duration_target(voiced: bool) -> AcousticRangeTarget {
    let (min, max) = if voiced { (40.0, 110.0) } else { (50.0, 130.0) };
    range_target(
        AcousticMeasurement::ClosureDuration,
        min,
        max,
        "ms",
        0.65,
        Some(
            "Broad oral stop closure duration range; prosodic position can stretch or compress it.",
        ),
    )
}

fn frication_targets(spectral_shape: &str, voicing: &str) -> Vec<AcousticRangeTarget> {
    let duration = if voicing == "voiced" {
        (45.0, 150.0)
    } else {
        (60.0, 190.0)
    };
    let centroid = match spectral_shape {
        "high_sibilant" => (4500.0, 8500.0),
        "lower_sibilant" => (2500.0, 6500.0),
        "diffuse_labiodental" => (2500.0, 7000.0),
        "diffuse_dental" => (3000.0, 8000.0),
        "diffuse_glottal" => (1000.0, 5000.0),
        _ => (2000.0, 6500.0),
    };
    let skew = match spectral_shape {
        "high_sibilant" => (0.6, 1.4),
        "lower_sibilant" => (-0.2, 0.6),
        "diffuse_labiodental" => (-0.6, 0.3),
        "diffuse_dental" => (-0.3, 0.5),
        "diffuse_glottal" => (-1.0, 0.2),
        _ => (-0.5, 0.5),
    };
    vec![
        range_target(
            AcousticMeasurement::FricationDuration,
            duration.0,
            duration.1,
            "ms",
            0.65,
            Some(
                "Sustained frication interval; voiced fricatives often shorten in running speech.",
            ),
        ),
        range_target(
            AcousticMeasurement::SpectralCentroid,
            centroid.0,
            centroid.1,
            "Hz",
            0.6,
            Some("Coarse frication centroid band keyed by English fricative place and sibilance."),
        ),
        range_target(
            AcousticMeasurement::SpectralSkew,
            skew.0,
            skew.1,
            "unitless",
            0.55,
            Some("Coarse spectral skew cue; most useful for separating [s] from [ʃ]."),
        ),
    ]
}

fn frication_spectral_skew(spectral_shape: &str) -> &'static str {
    match spectral_shape {
        "high_sibilant" => "strong_high_frequency_skew",
        "lower_sibilant" => "moderate_postalveolar_skew",
        "diffuse_labiodental" => "weak_diffuse_labiodental_skew",
        "diffuse_dental" => "weak_diffuse_dental_skew",
        "diffuse_glottal" => "weak_breathy_glottal_skew",
        _ => "diffuse_skew",
    }
}

fn frication_strength(spectral_shape: &str) -> &'static str {
    match spectral_shape {
        "high_sibilant" => "strong_anterior_sibilant",
        "lower_sibilant" => "strong_postalveolar_sibilant",
        "diffuse_labiodental" => "weak_diffuse_labiodental",
        "diffuse_dental" => "weak_diffuse_dental",
        "diffuse_glottal" => "weak_diffuse_glottal",
        _ => "diffuse",
    }
}

fn frication_skew_weight(spectral_shape: &str) -> f32 {
    match spectral_shape {
        "high_sibilant" | "lower_sibilant" => 0.85,
        "diffuse_labiodental" | "diffuse_dental" | "diffuse_glottal" => 0.5,
        _ => 0.6,
    }
}

fn nasal_targets(place: &str) -> Vec<AcousticRangeTarget> {
    let (murmur, antiresonance, place_transition, antiresonance_note) = match place {
        "bilabial" => (
            (200.0, 350.0),
            (750.0, 1250.0),
            (800.0, 1300.0),
            "Bilabial nasal antiresonance often appears low in the spectrum.",
        ),
        "alveolar" | "dental" => (
            (250.0, 400.0),
            (1500.0, 2200.0),
            (1600.0, 2300.0),
            "Coronal nasal antiresonance is a mid-frequency place hint.",
        ),
        "velar" => (
            (250.0, 450.0),
            (2500.0, 3500.0),
            (1100.0, 2600.0),
            "Velar nasal antiresonance is often higher and strongly context-dependent.",
        ),
        _ => (
            (200.0, 450.0),
            (750.0, 3500.0),
            (800.0, 2600.0),
            "Nasal antiresonance place is unspecified; treat this only as an oral/nasal cue.",
        ),
    };
    vec![
        range_target(
            AcousticMeasurement::NasalMurmurBand,
            murmur.0,
            murmur.1,
            "Hz",
            0.65,
            Some("Low-frequency nasal murmur band for voiced nasal consonants."),
        ),
        range_target(
            AcousticMeasurement::NasalAntiresonance,
            antiresonance.0,
            antiresonance.1,
            "Hz",
            0.55,
            Some(antiresonance_note),
        ),
        range_target(
            AcousticMeasurement::NasalPlaceTransition,
            place_transition.0,
            place_transition.1,
            "Hz",
            0.5,
            Some("Coarse adjacent-vowel transition locus for nasal place."),
        ),
    ]
}

fn nasal_place_transition(place: &str) -> &'static str {
    match place {
        "bilabial" => "low_f2_labial_transition",
        "alveolar" | "dental" => "fronted_coronal_transition",
        "velar" => "velar_pinch_transition",
        _ => "unspecified_transition",
    }
}

fn affricate_closure_to_frication_target(voiced: bool) -> AcousticRangeTarget {
    let (min, max) = if voiced { (5.0, 30.0) } else { (8.0, 35.0) };
    range_target(
        AcousticMeasurement::AffricateClosureToFrication,
        min,
        max,
        "ms",
        0.65,
        Some("Short release-to-sustained-frication interval for English postalveolar affricates."),
    )
}

fn add_fricative_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    range_targets: &mut Vec<AcousticRangeTarget>,
    spectral_shape: &str,
    voicing: &str,
) {
    put_acoustic_feature(
        features,
        "frication_noise",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "frication_spectral_shape",
        Spec::Known(FeatureValue::Category(spectral_shape.into())),
    );
    put_acoustic_feature(
        features,
        "frication_spectral_skew",
        Spec::Known(FeatureValue::Category(
            frication_spectral_skew(spectral_shape).into(),
        )),
    );
    put_acoustic_feature(
        features,
        "frication_strength",
        Spec::Known(FeatureValue::Category(
            frication_strength(spectral_shape).into(),
        )),
    );
    cues.extend(weighted_cues(&[
        ("acoustic.cue.frication_noise", 1.0),
        ("acoustic.cue.frication_spectral_shape", 0.9),
        (
            "acoustic.cue.frication_spectral_skew",
            frication_skew_weight(spectral_shape),
        ),
    ]));
    let frication_ranges = frication_targets(spectral_shape, voicing);
    range_targets.extend(frication_ranges.iter().cloned());
    landmarks.push(frication_landmark(
        "frication_noise",
        spectral_shape,
        &frication_ranges,
    ));
}

fn add_affricate_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    range_targets: &mut Vec<AcousticRangeTarget>,
    place: &str,
    voicing: &str,
    spectral_shape: &str,
) {
    let voiced = voicing == "voiced";
    let closure_voicing = if voiced {
        Spec::Variable(vec![FeatureValue::Bool(true), FeatureValue::Bool(false)])
    } else {
        Spec::Known(FeatureValue::Bool(false))
    };
    put_acoustic_feature(
        features,
        "stop_closure",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "release_burst",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(features, "voicing_during_closure", closure_voicing.clone());
    put_acoustic_feature(features, "vot_class", stop_vot_class(voiced));
    put_acoustic_feature(
        features,
        "stop_burst_spectral_shape",
        Spec::Known(FeatureValue::Category(
            stop_burst_spectral_shape(place).into(),
        )),
    );
    put_acoustic_feature(
        features,
        "aspiration_present",
        Spec::Known(FeatureValue::Bool(false)),
    );
    cues.extend(weighted_cues(&[
        ("acoustic.cue.stop_closure", 1.0),
        ("acoustic.cue.release_burst", 0.8),
        ("acoustic.cue.stop_burst_spectral_shape", 0.7),
        ("acoustic.cue.voice_onset_time", 0.5),
        (
            "acoustic.cue.closure_voicing",
            if voiced { 0.7 } else { 0.4 },
        ),
    ]));
    let closure_duration = stop_closure_duration_target(voiced);
    range_targets.push(stop_vot_target(place, voiced));
    range_targets.push(closure_duration.clone());

    landmarks.push(closure_landmark(closure_voicing, &[closure_duration]));
    landmarks.push(release_burst_landmark(stop_burst_spectral_shape(place)));

    add_fricative_acoustics(
        features,
        cues,
        landmarks,
        range_targets,
        spectral_shape,
        voicing,
    );
    put_acoustic_feature(
        features,
        "affricate_release",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "affricate_closure_to_frication_timing",
        Spec::Known(FeatureValue::Category(
            if voiced {
                "short_voiced_affricate_lag"
            } else {
                "short_voiceless_affricate_lag"
            }
            .into(),
        )),
    );
    cues.push(weighted_cue("acoustic.cue.affricate_release", 1.0));
    cues.push(weighted_cue(
        "acoustic.cue.affricate_closure_to_frication_timing",
        0.85,
    ));
    let affricate_timing = affricate_closure_to_frication_target(voiced);
    range_targets.push(affricate_timing.clone());
    landmarks.push(affricate_release_landmark(&[affricate_timing]));
    landmarks.push(voicing_onset_landmark(
        voiced,
        Spec::Known(FeatureValue::Bool(false)),
    ));
}

fn add_nasal_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    range_targets: &mut Vec<AcousticRangeTarget>,
    place: &str,
) {
    put_acoustic_feature(
        features,
        "nasal_murmur",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "nasal_antiresonance",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "nasal_place",
        Spec::Known(FeatureValue::Category(nasal_place_cue(place).into())),
    );
    put_acoustic_feature(
        features,
        "nasal_place_transition",
        Spec::Known(FeatureValue::Category(nasal_place_transition(place).into())),
    );
    cues.extend(weighted_cues(&[
        ("acoustic.cue.nasal_murmur", 1.0),
        ("acoustic.cue.nasal_antiresonance", 0.8),
        ("acoustic.cue.nasal_place", 0.6),
        ("acoustic.cue.nasal_place_transition", 0.65),
        ("acoustic.cue.periodic_voicing", 0.9),
    ]));
    let nasal_ranges = nasal_targets(place);
    range_targets.extend(nasal_ranges.iter().cloned());
    landmarks.push(nasal_murmur_landmark(nasal_place_cue(place), &nasal_ranges));
}

fn add_approximant_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    range_targets: &mut Vec<AcousticRangeTarget>,
    manner: &str,
    trajectory: &str,
    rhotic: bool,
    lateral: bool,
    l_quality: Option<&str>,
    label: &str,
) {
    put_acoustic_feature(
        features,
        "approximant_formants",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "formant_trajectory",
        Spec::Known(FeatureValue::Category(trajectory.into())),
    );
    if rhotic {
        put_acoustic_feature(
            features,
            "f3_region",
            Spec::Known(FeatureValue::Category("low".into())),
        );
    }
    if lateral {
        put_acoustic_feature(
            features,
            "lateral_resonance",
            Spec::Known(FeatureValue::Bool(true)),
        );
    }
    if let Some(l_quality) = l_quality {
        put_acoustic_feature(
            features,
            "l_quality",
            Spec::Known(FeatureValue::Category(l_quality.into())),
        );
    }
    put_acoustic_feature(
        features,
        "approximant_formant_transition_detail",
        Spec::Known(FeatureValue::Category(
            approximant_transition_detail(trajectory).into(),
        )),
    );

    cues.extend(weighted_cues(&[
        ("acoustic.cue.approximant_formants", 1.0),
        ("acoustic.cue.formant_trajectory", 0.8),
        ("acoustic.cue.approximant_formant_transition_detail", 0.85),
        ("acoustic.cue.periodic_voicing", 0.9),
    ]));
    if rhotic {
        cues.push(weighted_cue("acoustic.cue.f3_region", 0.8));
    }
    let transition_ranges = approximant_transition_targets(trajectory);
    range_targets.extend(transition_ranges.iter().cloned());
    landmarks.push(approximant_landmark(
        manner,
        trajectory,
        label,
        &transition_ranges,
    ));
}

fn add_tap_acoustics(
    features: &mut FeatureBundle,
    cues: &mut Vec<WeightedCue>,
    landmarks: &mut Vec<AcousticLandmark>,
    range_targets: &mut Vec<AcousticRangeTarget>,
) {
    put_acoustic_feature(
        features,
        "tap_closure",
        Spec::Known(FeatureValue::Bool(true)),
    );
    put_acoustic_feature(
        features,
        "aspiration_present",
        Spec::Known(FeatureValue::Bool(false)),
    );
    cues.extend(weighted_cues(&[
        ("acoustic.cue.tap_closure", 1.0),
        ("acoustic.cue.periodic_voicing", 0.8),
        ("acoustic.cue.consonant_place_transition", 0.6),
    ]));
    let closure_duration = range_target(
        AcousticMeasurement::ClosureDuration,
        10.0,
        30.0,
        "ms",
        0.75,
        Some("Intervocalic English taps are brief closures, not full oral stop holds."),
    );
    range_targets.push(closure_duration.clone());
    landmarks.push(tap_closure_landmark(&[closure_duration]));
}

fn frication_spectral_shape_for_entry(entry: &arpabet::ArpabetEntry) -> &'static str {
    match (entry.place, entry.symbol) {
        (Some("alveolar"), "S" | "Z") => "high_sibilant",
        (Some("postalveolar"), _) => "lower_sibilant",
        (Some("labiodental"), _) => "diffuse_labiodental",
        (Some("dental"), _) => "diffuse_dental",
        (Some("glottal"), _) => "diffuse_glottal",
        _ => "diffuse",
    }
}

fn frication_spectral_shape_for_place(place: &str) -> &'static str {
    match place {
        "alveolar" => "high_sibilant",
        "postalveolar" => "lower_sibilant",
        "labiodental" => "diffuse_labiodental",
        "dental" => "diffuse_dental",
        "glottal" => "diffuse_glottal",
        _ => "diffuse",
    }
}

fn approximant_trajectory_for_entry(entry: &arpabet::ArpabetEntry) -> &'static str {
    match entry.symbol {
        "W" => "velar_labial_glide",
        "Y" => "palatal_glide",
        "L" => "lateral_approximant",
        "R" => "rhotic_approximant",
        _ => "smooth_approximant",
    }
}

fn approximant_transition_detail(trajectory: &str) -> &'static str {
    match trajectory {
        "velar_labial_glide" => "low_f2_rounded_glide",
        "palatal_glide" => "high_f2_palatal_glide",
        "rhotic_approximant" => "lowered_f3_rhotic_transition",
        "lateral_approximant" => "coronal_lateral_f2_transition",
        "velarized_lateral_approximant" => "dark_l_low_f2_velarized_transition",
        _ => "smooth_voiced_transition",
    }
}

fn approximant_transition_targets(trajectory: &str) -> Vec<AcousticRangeTarget> {
    match trajectory {
        "velar_labial_glide" => vec![
            formant_transition_target(
                2,
                -900.0,
                -250.0,
                0.65,
                "W has a low-to-lower F2 glide shaped by velar and labial constriction.",
            ),
            formant_transition_target(
                3,
                -500.0,
                100.0,
                0.45,
                "W can lower upper formants, but F3 is less stable than F2.",
            ),
        ],
        "palatal_glide" => vec![
            formant_transition_target(
                2,
                500.0,
                1400.0,
                0.7,
                "Y is cued by a strong high-F2 palatal transition.",
            ),
            formant_transition_target(
                3,
                100.0,
                600.0,
                0.45,
                "Y often raises upper formants with substantial speaker variation.",
            ),
        ],
        "rhotic_approximant" => vec![
            formant_transition_target(
                3,
                -1200.0,
                -350.0,
                0.75,
                "English R is cued by a strong F3 lowering transition.",
            ),
            formant_transition_target(
                2,
                -300.0,
                300.0,
                0.45,
                "R has variable F2 movement depending on context.",
            ),
        ],
        "lateral_approximant" => vec![
            formant_transition_target(
                2,
                100.0,
                700.0,
                0.55,
                "Light L tends to show a fronted coronal F2 transition.",
            ),
            formant_transition_target(
                3,
                -200.0,
                300.0,
                0.4,
                "Light L upper-formant movement is weaker than the F2 cue.",
            ),
        ],
        "velarized_lateral_approximant" => vec![
            formant_transition_target(
                2,
                -700.0,
                -100.0,
                0.6,
                "Dark L has a lowered F2 transition from velarization.",
            ),
            formant_transition_target(
                1,
                50.0,
                350.0,
                0.45,
                "Dark L can raise F1 relative to light L.",
            ),
        ],
        _ => Vec::new(),
    }
}

fn formant_transition_target(
    index: u8,
    min: f32,
    max: f32,
    confidence: f32,
    notes: &str,
) -> AcousticRangeTarget {
    range_target(
        AcousticMeasurement::FormantTransition { index },
        min,
        max,
        "Hz_delta",
        confidence,
        Some(notes),
    )
}

fn vowel_target_landmark(range_targets: &[AcousticRangeTarget]) -> AcousticLandmark {
    AcousticLandmark {
        id: "steady_vowel_target".into(),
        kind: AcousticLandmarkKind::VowelTarget,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.04,
            end_s: 0.04,
        },
        expected_features: FeatureBundle::default(),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.f1_region", 0.8),
            ("acoustic.cue.f2_region", 1.0),
        ]),
        range_targets: range_targets.to_vec(),
        notes: Some(
            "Sample formants near the steady middle of the vowel when the segment is long enough."
                .into(),
        ),
    }
}

fn syllable_nucleus_landmark() -> AcousticLandmark {
    AcousticLandmark {
        id: "syllable_nucleus_peak".into(),
        kind: AcousticLandmarkKind::VowelTarget,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.06,
            end_s: 0.06,
        },
        expected_features: acoustic_feature_bundle(&[
            ("vowel_nucleus", Spec::Known(FeatureValue::Bool(true))),
            ("periodic_voicing", Spec::Known(FeatureValue::Bool(true))),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.vowel_nucleus", 1.0),
            ("acoustic.cue.sonority_peak", 0.9),
            ("acoustic.cue.periodic_voicing", 0.8),
        ]),
        range_targets: Vec::new(),
        notes: Some("Use this as the preferred anchor when fitting syllable timing.".into()),
    }
}

fn formant_trajectory_landmark(trajectory: &str) -> AcousticLandmark {
    AcousticLandmark {
        id: "formant_trajectory".into(),
        kind: AcousticLandmarkKind::FormantTransition,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.08,
            end_s: 0.08,
        },
        expected_features: acoustic_feature_bundle(&[(
            "formant_trajectory",
            Spec::Known(FeatureValue::Category(trajectory.into())),
        )]),
        weighted_cues: weighted_cues(&[("acoustic.cue.formant_trajectory", 1.0)]),
        range_targets: Vec::new(),
        notes: Some(
            "Fit the direction of F1/F2 movement across the vowel, not just the midpoint.".into(),
        ),
    }
}

fn rhotic_target_landmark() -> AcousticLandmark {
    AcousticLandmark {
        id: "rhotic_target".into(),
        kind: AcousticLandmarkKind::VowelTarget,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.05,
            end_s: 0.05,
        },
        expected_features: acoustic_feature_bundle(&[(
            "f3_region",
            Spec::Known(FeatureValue::Category("low".into())),
        )]),
        weighted_cues: weighted_cues(&[("acoustic.cue.f3_region", 1.0)]),
        range_targets: vec![range_target(
            AcousticMeasurement::Formant { index: 3 },
            1300.0,
            2200.0,
            "Hz",
            0.7,
            Some("English rhotic targets lower F3 relative to non-rhotic vowels."),
        )],
        notes: Some(
            "English r-colored vowels are expected to show a lowered third formant.".into(),
        ),
    }
}

fn closure_landmark(
    voicing_during_closure: Spec<FeatureValue>,
    range_targets: &[AcousticRangeTarget],
) -> AcousticLandmark {
    AcousticLandmark {
        id: "stop_closure".into(),
        kind: AcousticLandmarkKind::Closure,
        anchor: LandmarkAnchor::Release,
        window: RelativeTimeWindow {
            start_s: -0.08,
            end_s: 0.0,
        },
        expected_features: acoustic_feature_bundle(&[(
            "voicing_during_closure",
            voicing_during_closure,
        )]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.stop_closure", 1.0),
            ("acoustic.cue.closure_voicing", 0.8),
        ]),
        range_targets: range_targets.to_vec(),
        notes: None,
    }
}

fn frication_landmark(
    id: &str,
    spectral_shape: &str,
    range_targets: &[AcousticRangeTarget],
) -> AcousticLandmark {
    AcousticLandmark {
        id: id.into(),
        kind: AcousticLandmarkKind::AperiodicNoise,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.04,
            end_s: 0.04,
        },
        expected_features: acoustic_feature_bundle(&[
            ("frication_noise", Spec::Known(FeatureValue::Bool(true))),
            (
                "frication_spectral_shape",
                Spec::Known(FeatureValue::Category(spectral_shape.into())),
            ),
            (
                "frication_spectral_skew",
                Spec::Known(FeatureValue::Category(
                    frication_spectral_skew(spectral_shape).into(),
                )),
            ),
            (
                "frication_strength",
                Spec::Known(FeatureValue::Category(
                    frication_strength(spectral_shape).into(),
                )),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.frication_noise", 1.0),
            ("acoustic.cue.frication_spectral_shape", 0.9),
            (
                "acoustic.cue.frication_spectral_skew",
                frication_skew_weight(spectral_shape),
            ),
        ]),
        range_targets: range_targets.to_vec(),
        notes: Some("Track sustained aperiodic noise through the constriction interval.".into()),
    }
}

fn affricate_release_landmark(range_targets: &[AcousticRangeTarget]) -> AcousticLandmark {
    AcousticLandmark {
        id: "affricate_release".into(),
        kind: AcousticLandmarkKind::ReleaseBurst,
        anchor: LandmarkAnchor::Release,
        window: RelativeTimeWindow {
            start_s: -0.005,
            end_s: 0.05,
        },
        expected_features: acoustic_feature_bundle(&[
            ("affricate_release", Spec::Known(FeatureValue::Bool(true))),
            (
                "affricate_closure_to_frication_timing",
                Spec::Known(FeatureValue::Category("short_affricate_lag".into())),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.release_burst", 0.8),
            ("acoustic.cue.affricate_release", 1.0),
            ("acoustic.cue.affricate_closure_to_frication_timing", 0.85),
            ("acoustic.cue.frication_noise", 0.8),
        ]),
        range_targets: range_targets.to_vec(),
        notes: Some("Affricate release should include a stop burst followed by frication.".into()),
    }
}

fn nasal_murmur_landmark(
    place_cue: &str,
    range_targets: &[AcousticRangeTarget],
) -> AcousticLandmark {
    let transition = match place_cue {
        "labial_murmur" => "low_f2_labial_transition",
        "coronal_murmur" => "fronted_coronal_transition",
        "velar_murmur" => "velar_pinch_transition",
        _ => "unspecified_transition",
    };
    AcousticLandmark {
        id: "nasal_murmur".into(),
        kind: AcousticLandmarkKind::PeriodicVoicing,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.05,
            end_s: 0.05,
        },
        expected_features: acoustic_feature_bundle(&[
            ("nasal_murmur", Spec::Known(FeatureValue::Bool(true))),
            ("nasal_antiresonance", Spec::Known(FeatureValue::Bool(true))),
            (
                "nasal_place",
                Spec::Known(FeatureValue::Category(place_cue.into())),
            ),
            (
                "nasal_place_transition",
                Spec::Known(FeatureValue::Category(transition.into())),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.nasal_murmur", 1.0),
            ("acoustic.cue.nasal_antiresonance", 0.8),
            ("acoustic.cue.nasal_place_transition", 0.65),
            ("acoustic.cue.periodic_voicing", 0.8),
        ]),
        range_targets: range_targets.to_vec(),
        notes: Some(
            "Use nasal murmur and antiresonance cues for nasal consonant alignment.".into(),
        ),
    }
}

fn approximant_landmark(
    manner: &str,
    trajectory: &str,
    label: &str,
    range_targets: &[AcousticRangeTarget],
) -> AcousticLandmark {
    AcousticLandmark {
        id: format!("{manner}_approximant_transition"),
        kind: AcousticLandmarkKind::FormantTransition,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.06,
            end_s: 0.06,
        },
        expected_features: acoustic_feature_bundle(&[
            (
                "approximant_formants",
                Spec::Known(FeatureValue::Bool(true)),
            ),
            (
                "formant_trajectory",
                Spec::Known(FeatureValue::Category(trajectory.into())),
            ),
            (
                "approximant_formant_transition_detail",
                Spec::Known(FeatureValue::Category(
                    approximant_transition_detail(trajectory).into(),
                )),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.approximant_formants", 1.0),
            ("acoustic.cue.formant_trajectory", 0.8),
            ("acoustic.cue.approximant_formant_transition_detail", 0.85),
            ("acoustic.cue.periodic_voicing", 0.8),
        ]),
        range_targets: range_targets.to_vec(),
        notes: Some(format!(
            "Track smooth voiced formant movement for English {manner} {label}."
        )),
    }
}

fn tap_closure_landmark(range_targets: &[AcousticRangeTarget]) -> AcousticLandmark {
    AcousticLandmark {
        id: "brief_tap_closure".into(),
        kind: AcousticLandmarkKind::Closure,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.015,
            end_s: 0.015,
        },
        expected_features: acoustic_feature_bundle(&[(
            "tap_closure",
            Spec::Known(FeatureValue::Bool(true)),
        )]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.tap_closure", 1.0),
            ("acoustic.cue.periodic_voicing", 0.6),
        ]),
        range_targets: range_targets.to_vec(),
        notes: Some(
            "A tap closure should be brief and voiced compared with a full oral stop.".into(),
        ),
    }
}

fn release_burst_landmark(spectral_shape: &str) -> AcousticLandmark {
    AcousticLandmark {
        id: "release_burst".into(),
        kind: AcousticLandmarkKind::ReleaseBurst,
        anchor: LandmarkAnchor::Release,
        window: RelativeTimeWindow {
            start_s: -0.005,
            end_s: 0.02,
        },
        expected_features: acoustic_feature_bundle(&[
            ("release_burst", Spec::Known(FeatureValue::Bool(true))),
            (
                "stop_burst_spectral_shape",
                Spec::Known(FeatureValue::Category(spectral_shape.into())),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.release_burst", 1.0),
            ("acoustic.cue.stop_burst_spectral_shape", 0.8),
        ]),
        range_targets: Vec::new(),
        notes: Some("Burst timing is a useful alignment point for oral stops.".into()),
    }
}

fn boundary_landmark(boundary_kind: &str, gap_class: &str, silent: bool) -> AcousticLandmark {
    AcousticLandmark {
        id: format!("{boundary_kind}_boundary"),
        kind: AcousticLandmarkKind::Boundary,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.005,
            end_s: 0.005,
        },
        expected_features: acoustic_feature_bundle(&[
            (
                "segment_boundary",
                Spec::Known(FeatureValue::Category(boundary_kind.into())),
            ),
            (
                "boundary_gap",
                Spec::Known(FeatureValue::Category(gap_class.into())),
            ),
            ("silent_boundary", Spec::Known(FeatureValue::Bool(silent))),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.segment_boundary", 1.0),
            ("acoustic.cue.boundary_gap", if silent { 0.9 } else { 0.25 }),
        ]),
        range_targets: boundary_range_targets(gap_class),
        notes: Some(if silent {
            "Boundary expects an audible silent gap class.".into()
        } else {
            "Boundary is an alignment anchor and should not require audible silence.".into()
        }),
    }
}

fn aspiration_landmark() -> AcousticLandmark {
    AcousticLandmark {
        id: "post_release_aspiration".into(),
        kind: AcousticLandmarkKind::Aspiration,
        anchor: LandmarkAnchor::Release,
        window: RelativeTimeWindow {
            start_s: 0.01,
            end_s: 0.09,
        },
        expected_features: acoustic_feature_bundle(&[(
            "aspiration_present",
            Spec::Variable(vec![FeatureValue::Bool(false), FeatureValue::Bool(true)]),
        )]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.aspiration_noise", 1.0),
            ("acoustic.cue.voice_onset_time", 0.7),
        ]),
        range_targets: Vec::new(),
        notes: Some(
            "Search after release; English aspiration is conditioned by stress and position."
                .into(),
        ),
    }
}

fn voicing_onset_landmark(
    voiced: bool,
    aspiration_present: Spec<FeatureValue>,
) -> AcousticLandmark {
    AcousticLandmark {
        id: "voicing_onset".into(),
        kind: AcousticLandmarkKind::VoicingOnset,
        anchor: LandmarkAnchor::VoicingOnset,
        window: if voiced {
            RelativeTimeWindow {
                start_s: -0.03,
                end_s: 0.03,
            }
        } else {
            RelativeTimeWindow {
                start_s: 0.0,
                end_s: 0.12,
            }
        },
        expected_features: acoustic_feature_bundle(&[
            (
                "periodic_voicing",
                if voiced {
                    Spec::Variable(vec![FeatureValue::Bool(true), FeatureValue::Bool(false)])
                } else {
                    Spec::Known(FeatureValue::Bool(true))
                },
            ),
            ("aspiration_present", aspiration_present),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.voice_onset_time", 1.0),
            ("acoustic.cue.periodic_voicing", 0.8),
        ]),
        range_targets: Vec::new(),
        notes: Some(
            "Locate the transition into periodic voicing after closure/release cues.".into(),
        ),
    }
}

fn acoustic_feature_bundle(values: &[(&str, Spec<FeatureValue>)]) -> FeatureBundle {
    let mut bundle = FeatureBundle::default();
    for (name, value) in values {
        put_acoustic_feature(&mut bundle, name, value.clone());
    }
    bundle
}

fn put_acoustic_feature(bundle: &mut FeatureBundle, name: &str, value: Spec<FeatureValue>) {
    bundle
        .values
        .insert(FeatureId(format!("acoustic.{name}")), value);
}

fn phonology_category<'a>(features: &'a FeatureBundle, name: &str) -> Option<&'a str> {
    match features.values.get(&FeatureId(format!("phonology.{name}"))) {
        Some(Spec::Known(FeatureValue::Category(value))) => Some(value.as_str()),
        _ => None,
    }
}

fn phonology_bool(features: &FeatureBundle, name: &str) -> Option<bool> {
    match features.values.get(&FeatureId(format!("phonology.{name}"))) {
        Some(Spec::Known(FeatureValue::Bool(value))) => Some(*value),
        _ => None,
    }
}

fn weighted_cues(values: &[(&str, f32)]) -> Vec<WeightedCue> {
    values
        .iter()
        .map(|(cue, weight)| weighted_cue(cue, *weight))
        .collect()
}

fn weighted_cue(cue: &str, weight: f32) -> WeightedCue {
    WeightedCue {
        cue: AcousticCueId(cue.into()),
        weight,
    }
}

fn range_target(
    measurement: AcousticMeasurement,
    min: f32,
    max: f32,
    unit: &str,
    confidence: f32,
    notes: Option<&str>,
) -> AcousticRangeTarget {
    AcousticRangeTarget {
        measurement,
        range: NumericRange {
            min,
            max,
            unit: unit.into(),
        },
        confidence,
        notes: notes.map(str::to_owned),
    }
}

fn cue(
    id: &str,
    name: &str,
    feature: &str,
    targets: Vec<CueTarget>,
    notes: Option<String>,
) -> AcousticCueDef {
    AcousticCueDef {
        id: AcousticCueId(id.into()),
        name: name.into(),
        feature: FeatureId(feature.into()),
        targets,
        diagnosticity: cue_diagnosticity(id),
        dependencies: cue_dependencies(id),
        notes,
    }
}

fn cue_diagnosticity(id: &str) -> CueDiagnosticity {
    match id {
        "acoustic.cue.vowel_nucleus"
        | "acoustic.cue.f1_region"
        | "acoustic.cue.f2_region"
        | "acoustic.cue.stop_closure"
        | "acoustic.cue.release_burst"
        | "acoustic.cue.frication_noise"
        | "acoustic.cue.affricate_release"
        | "acoustic.cue.tap_closure"
        | "acoustic.cue.segment_boundary" => CueDiagnosticity::Robust,
        "acoustic.cue.rounding_resonance"
        | "acoustic.cue.f3_region"
        | "acoustic.cue.consonant_place_transition"
        | "acoustic.cue.place_formant_locus"
        | "acoustic.cue.frication_spectral_skew"
        | "acoustic.cue.nasal_place"
        | "acoustic.cue.nasal_place_transition"
        | "acoustic.cue.boundary_gap" => CueDiagnosticity::Weak,
        _ => CueDiagnosticity::Moderate,
    }
}

fn cue_dependencies(id: &str) -> Vec<CueDependency> {
    match id {
        "acoustic.cue.f1_region" | "acoustic.cue.f2_region" | "acoustic.cue.f3_region" => {
            vec![
                CueDependency::SpeakerDependent,
                CueDependency::ContextDependent,
            ]
        }
        "acoustic.cue.vowel_reduction"
        | "acoustic.cue.aspiration_noise"
        | "acoustic.cue.tap_closure"
        | "acoustic.cue.boundary_gap" => {
            vec![
                CueDependency::ContextDependent,
                CueDependency::StyleDependent,
            ]
        }
        "acoustic.cue.voice_onset_time"
        | "acoustic.cue.closure_voicing"
        | "acoustic.cue.frication_spectral_shape"
        | "acoustic.cue.frication_spectral_skew"
        | "acoustic.cue.nasal_place"
        | "acoustic.cue.nasal_place_transition"
        | "acoustic.cue.approximant_formant_transition_detail" => {
            vec![CueDependency::ContextDependent]
        }
        "acoustic.cue.sonority_peak" => vec![CueDependency::StyleDependent],
        _ => Vec::new(),
    }
}

fn allophone_rules(variety_id: &str) -> Vec<AllophoneRule> {
    let mut rules = Vec::new();

    for (symbol, phone) in [
        ("P", UNASPIRATED_P),
        ("T", UNASPIRATED_T),
        ("K", UNASPIRATED_K),
    ] {
        rules.push(AllophoneRule {
            id: format!(
                "american_english_unaspirated_{}_after_s",
                symbol.to_lowercase()
            ),
            name: format!("American English unaspirated /{symbol}/ after /s/"),
            input: phoneme_pattern(variety_id, symbol),
            environment: Environment {
                before: vec![SegmentMatcher::Phoneme(arpabet::phoneme_id(
                    variety_id, "S",
                ))],
                ..Default::default()
            },
            conditions: vec![RuleCondition::PreviousMatches(SegmentMatcher::Phoneme(
                arpabet::phoneme_id(variety_id, "S"),
            ))],
            output: PhonePattern {
                phone: Spec::Known(phone),
                features: feature_bundle(&[("aspiration", "unaspirated")]),
            },
            confidence: 0.95,
            status: RuleStatus::Productive,
        });
    }

    for (symbol, phone) in [("P", ASPIRATED_P), ("T", ASPIRATED_T), ("K", ASPIRATED_K)] {
        rules.push(AllophoneRule {
            id: format!(
                "american_english_aspirated_{}_stressed_onset",
                symbol.to_lowercase()
            ),
            name: format!("American English aspirated /{symbol}/ before a stressed vowel"),
            input: phoneme_pattern(variety_id, symbol),
            environment: Environment {
                after: vec![vowel_matcher()],
                ..Default::default()
            },
            conditions: vec![
                RuleCondition::NextMatches(vowel_matcher()),
                RuleCondition::NextStressIn(vec![Stress::Primary, Stress::Secondary]),
            ],
            output: PhonePattern {
                phone: Spec::Known(phone),
                features: feature_bundle(&[("aspiration", "aspirated")]),
            },
            confidence: 0.9,
            status: RuleStatus::Productive,
        });
    }

    for symbol in ["T", "D"] {
        let id = if symbol == "T" {
            "american_english_intervocalic_flapping".into()
        } else {
            "american_english_intervocalic_d_flapping".into()
        };
        rules.push(AllophoneRule {
            id,
            name: format!("American English intervocalic /{symbol}/ flapping"),
            input: phoneme_pattern(variety_id, symbol),
            environment: Environment {
                before: vec![vowel_matcher()],
                after: vec![vowel_matcher()],
                ..Default::default()
            },
            conditions: vec![
                RuleCondition::PreviousMatches(vowel_matcher()),
                RuleCondition::PreviousStressIn(vec![Stress::Primary, Stress::Secondary]),
                RuleCondition::NextMatches(vowel_matcher()),
                RuleCondition::NextStress(Stress::Unstressed),
                RuleCondition::NotCarefulStyle,
            ],
            output: PhonePattern {
                phone: Spec::Known(TAP),
                features: Default::default(),
            },
            confidence: 0.95,
            status: RuleStatus::StyleDependent,
        });
    }

    rules.push(AllophoneRule {
        id: "american_english_light_l_before_vowel".into(),
        name: "American English light /l/ before vowels".into(),
        input: phoneme_pattern(variety_id, "L"),
        environment: Environment {
            after: vec![vowel_matcher()],
            ..Default::default()
        },
        conditions: vec![RuleCondition::NextMatches(vowel_matcher())],
        output: PhonePattern {
            phone: Spec::Known(L),
            features: feature_bundle(&[("l_quality", "light")]),
        },
        confidence: 0.85,
        status: RuleStatus::Productive,
    });
    rules.push(AllophoneRule {
        id: "american_english_dark_l_elsewhere".into(),
        name: "American English dark /l/ outside prevocalic light-l contexts".into(),
        input: phoneme_pattern(variety_id, "L"),
        environment: Environment::default(),
        conditions: Vec::new(),
        output: PhonePattern {
            phone: Spec::Known(DARK_L),
            features: feature_bundle(&[("l_quality", "dark")]),
        },
        confidence: 0.75,
        status: RuleStatus::Productive,
    });

    for (stress, label) in [
        (Stress::Primary, "primary"),
        (Stress::Secondary, "secondary"),
    ] {
        for (symbol, phone, allophone_name) in [
            ("AH", STRUT, "strut"),
            ("ER", STRESSED_RHOTIC_VOWEL, "stressed_rhotic"),
        ] {
            rules.push(AllophoneRule {
                id: format!(
                    "american_english_stressed_{}_{label}_{allophone_name}_allophone",
                    symbol.to_lowercase()
                ),
                name: format!(
                    "American English {label}-stressed /{symbol}/ {allophone_name} allophone"
                ),
                input: phoneme_pattern(variety_id, symbol),
                environment: Environment {
                    syllable_position: Spec::Known(SyllablePosition::Nucleus),
                    stress_context: Spec::Known(stress.clone()),
                    ..Default::default()
                },
                conditions: Vec::new(),
                output: PhonePattern {
                    phone: Spec::Known(phone),
                    features: feature_bundle_with_values(&[(
                        "phonology.stress_conditioned_allophone",
                        FeatureValue::Bool(true),
                    )]),
                },
                confidence: 0.95,
                status: RuleStatus::Productive,
            });
        }
    }

    for (symbol, phone) in [("AH", SCHWA), ("ER", R_COLORED_SCHWA)] {
        rules.push(AllophoneRule {
            id: format!(
                "american_english_unstressed_{}_nucleus_reduction",
                symbol.to_lowercase()
            ),
            name: format!("American English unstressed /{symbol}/ nucleus reduction"),
            input: phoneme_pattern(variety_id, symbol),
            environment: Environment {
                syllable_position: Spec::Known(SyllablePosition::Nucleus),
                stress_context: Spec::Known(Stress::Unstressed),
                ..Default::default()
            },
            conditions: Vec::new(),
            output: PhonePattern {
                phone: Spec::Known(phone),
                features: feature_bundle_with_values(&[
                    ("phonology.reduced_vowel", FeatureValue::Bool(true)),
                    (
                        "phonology.reduction_context",
                        FeatureValue::Category("unstressed_nucleus".into()),
                    ),
                ]),
            },
            confidence: 0.95,
            status: RuleStatus::Productive,
        });
    }

    for symbol in ["B", "D", "G", "V", "DH", "Z", "ZH", "JH"] {
        for word_position in [WordPosition::Final, WordPosition::Isolated] {
            let position_id = match word_position {
                WordPosition::Final => "word_final",
                WordPosition::Isolated => "isolated",
                _ => unreachable!("final devoicing positions are explicit"),
            };
            rules.push(AllophoneRule {
                id: format!(
                    "american_english_optional_final_devoicing_{}_{}",
                    symbol.to_lowercase(),
                    position_id
                ),
                name: format!(
                    "American English optional final devoicing of /{symbol}/ in {position_id} position"
                ),
                input: phoneme_pattern(variety_id, symbol),
                environment: Environment {
                    word_position: Spec::Known(word_position),
                    ..Default::default()
                },
                conditions: Vec::new(),
                output: PhonePattern {
                    phone: Spec::Known(arpabet_phone_id(symbol)),
                    features: feature_bundle_with_values(&[
                        ("phonology.partial_devoicing", FeatureValue::Bool(true)),
                        (
                            "phonology.devoicing",
                            FeatureValue::Category("final_optional".into()),
                        ),
                    ]),
                },
                confidence: 0.6,
                status: RuleStatus::Optional,
            });
        }
    }

    for symbol in ["B", "D", "G", "V", "DH", "Z", "ZH", "JH"] {
        rules.push(AllophoneRule {
            id: format!(
                "american_english_partial_devoicing_{}_before_voiceless_obstruent",
                symbol.to_lowercase()
            ),
            name: format!(
                "American English partial devoicing of /{symbol}/ before voiceless obstruents"
            ),
            input: phoneme_pattern(variety_id, symbol),
            environment: Environment {
                after: vec![voiceless_consonant_matcher()],
                ..Default::default()
            },
            conditions: vec![RuleCondition::NextMatches(voiceless_consonant_matcher())],
            output: PhonePattern {
                phone: Spec::Known(arpabet_phone_id(symbol)),
                features: feature_bundle_with_values(&[
                    ("phonology.partial_devoicing", FeatureValue::Bool(true)),
                    (
                        "phonology.devoicing",
                        FeatureValue::Category("partial".into()),
                    ),
                ]),
            },
            confidence: 0.65,
            status: RuleStatus::Optional,
        });
    }

    rules.push(AllophoneRule {
        id: "alveolar_nasal_velar_assimilation".into(),
        name: "Alveolar nasal velar assimilation".into(),
        input: phoneme_pattern(variety_id, "N"),
        environment: Environment {
            after: vec![SegmentMatcher::FeatureBundle(feature_bundle(&[
                ("place", "velar"),
                ("manner", "stop"),
            ]))],
            ..Default::default()
        },
        conditions: vec![RuleCondition::NextMatches(SegmentMatcher::FeatureBundle(
            feature_bundle(&[("place", "velar"), ("manner", "stop")]),
        ))],
        output: PhonePattern {
            phone: Spec::Known(NG),
            features: Default::default(),
        },
        confidence: 0.98,
        status: RuleStatus::Productive,
    });

    rules
}

fn phoneme_pattern(variety_id: &str, symbol: &str) -> PhonemePattern {
    PhonemePattern {
        phoneme: Spec::Known(arpabet::phoneme_id(variety_id, symbol)),
        features: Default::default(),
    }
}

fn vowel_matcher() -> SegmentMatcher {
    SegmentMatcher::FeatureBundle(feature_bundle(&[("major", "vowel")]))
}

fn voiceless_consonant_matcher() -> SegmentMatcher {
    SegmentMatcher::FeatureBundle(feature_bundle(&[
        ("major", "consonant"),
        ("voicing", "voiceless"),
    ]))
}

fn arpabet_phone_id(symbol: &str) -> PhoneId {
    let entry = arpabet::entry(symbol).expect("known English ARPABET symbol");
    arpabet::phone_id_for_ipa(entry.phone_symbol)
}

fn epenthesis_rules() -> Vec<EpenthesisRule> {
    vec![EpenthesisRule {
        id: "english_letter_name_front_vowel_linking_yod".into(),
        name: "English letter-name front-vowel linking yod".into(),
        before: vec![SegmentMatcher::FeatureBundle(feature_bundle_with_values(
            &[
                ("phonology.major", FeatureValue::Category("vowel".into())),
                (
                    "phonology.vowel_backness",
                    FeatureValue::Category("front".into()),
                ),
                ("orthography.letter_name", FeatureValue::Bool(true)),
            ],
        ))],
        after: vec![SegmentMatcher::FeatureBundle(feature_bundle_with_values(
            &[
                ("phonology.major", FeatureValue::Category("vowel".into())),
                ("orthography.letter_name", FeatureValue::Bool(true)),
            ],
        ))],
        output: PhonePattern {
            phone: Spec::Known(Y),
            features: Default::default(),
        },
        confidence: 0.85,
        status: RuleStatus::Productive,
    }]
}

fn feature_bundle(values: &[(&str, &str)]) -> FeatureBundle {
    let mut bundle = FeatureBundle::default();
    for (name, value) in values {
        bundle.values.insert(
            FeatureId(format!("phonology.{name}")),
            Spec::Known(FeatureValue::Category((*value).into())),
        );
    }
    bundle
}

fn feature_bundle_with_values(values: &[(&str, FeatureValue)]) -> FeatureBundle {
    let mut bundle = FeatureBundle::default();
    for (id, value) in values {
        bundle
            .values
            .insert(FeatureId((*id).into()), Spec::Known(value.clone()));
    }
    bundle
}

fn phonotactics(singing: bool) -> Phonotactics {
    let mut constraints = Vec::new();
    constraints.push(PhonotacticConstraint {
        id: "english.illegal_onset.ng".into(),
        description: "Velar nasal is not a legal singleton onset in English".into(),
        matcher: SegmentMatcher::Phone(NG),
        environment: Environment {
            syllable_position: Spec::Known(crate::segment::SyllablePosition::Onset),
            ..Default::default()
        },
        status: RuleStatus::Productive,
    });

    for cluster in LEGAL_ONSETS {
        constraints.push(cluster_constraint(
            ClusterScope::Onset,
            cluster,
            RuleStatus::Productive,
        ));
    }
    if singing {
        for cluster in SINGING_ONSET_ADDITIONS {
            constraints.push(cluster_constraint(
                ClusterScope::SingingOnset,
                cluster,
                RuleStatus::Experimental,
            ));
        }
    }
    for cluster in LEGAL_CODAS {
        constraints.push(cluster_constraint(
            ClusterScope::Coda,
            cluster,
            RuleStatus::Productive,
        ));
    }

    Phonotactics {
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
            SyllableShape {
                pattern: "CCVC".into(),
            },
            SyllableShape {
                pattern: "CVCC".into(),
            },
        ],
        constraints,
    }
}

fn cluster_constraint(
    scope: ClusterScope,
    cluster: &[PhoneId],
    status: RuleStatus,
) -> PhonotacticConstraint {
    let suffix = cluster_suffix(cluster);
    let label = cluster_label(cluster);
    PhonotacticConstraint {
        id: format!("{}.{}", scope.constraint_prefix(), suffix),
        description: format!("Legal {} cluster {}", scope.label(), label),
        matcher: SegmentMatcher::Any,
        environment: Environment {
            before: cluster.iter().cloned().map(SegmentMatcher::Phone).collect(),
            syllable_position: Spec::Known(scope.syllable_position()),
            prosodic_context: scope.prosodic_context(),
            ..Default::default()
        },
        status,
    }
}

impl ClusterScope {
    fn constraint_prefix(self) -> &'static str {
        match self {
            Self::Onset => "english.legal_onset",
            Self::SingingOnset => "english.singing_legal_onset",
            Self::Coda => "english.legal_coda",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Onset | Self::SingingOnset => "onset",
            Self::Coda => "coda",
        }
    }

    fn syllable_position(self) -> SyllablePosition {
        match self {
            Self::Onset | Self::SingingOnset => SyllablePosition::Onset,
            Self::Coda => SyllablePosition::Coda,
        }
    }

    fn prosodic_context(self) -> Spec<ProsodicContext> {
        match self {
            Self::SingingOnset => Spec::Known(ProsodicContext::Emphasized),
            Self::Onset | Self::Coda => Spec::Unspecified,
        }
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
    use crate::ids::PhonemeId;

    fn has_cluster(variety: &LinguisticVariety, needle: &str) -> bool {
        variety
            .phonotactics
            .as_ref()
            .unwrap()
            .constraints
            .iter()
            .any(|constraint| constraint.id.ends_with(needle))
    }

    #[test]
    fn singing_adds_tl_without_changing_ga() {
        assert!(!has_cluster(&variety("en-US-GA"), "t_l"));
        assert!(has_cluster(&variety("en-US-singing"), "t_l"));
    }

    #[test]
    fn ga_inventory_contains_canonical_phonemes_and_ipa_phones() {
        let ga = variety("en-US-GA");
        assert!(
            ga.phonemes
                .phonemes
                .contains_key(&PhonemeId("en-US-GA.phoneme.ʌ".into()))
        );
        assert!(ga.phones.phones.contains_key(&PhoneId::from("ipa.phone.ʌ")));
        assert_eq!(
            ga.phones.phones.get(&SCHWA).expect("schwa phone").status,
            crate::segment::SegmentStatus::Core
        );
        assert_eq!(
            ga.phones.phones.get(&STRUT).expect("strut phone").status,
            crate::segment::SegmentStatus::Allophonic
        );
        assert_eq!(
            ga.phones
                .phones
                .get(&R_COLORED_SCHWA)
                .expect("r-colored schwa phone")
                .status,
            crate::segment::SegmentStatus::Core
        );
        assert_eq!(
            ga.phones
                .phones
                .get(&STRESSED_RHOTIC_VOWEL)
                .expect("stressed rhotic vowel phone")
                .status,
            crate::segment::SegmentStatus::Allophonic
        );
        let ah = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "AH"))
            .expect("AH phoneme decoded to canonical id");
        assert_eq!(ah.id, PhonemeId("en-US-GA.phoneme.ʌ".into()));
        assert_eq!(ah.default_phone, Some(SCHWA));
        assert!(
            ah.aliases
                .iter()
                .any(|alias| alias.system == "arpabet" && alias.symbol == "AH")
        );
        let er = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "ER"))
            .expect("ER phoneme decoded to canonical id");
        assert_eq!(er.id, PhonemeId("en-US-GA.phoneme.ɝ".into()));
        assert_eq!(er.default_phone, Some(R_COLORED_SCHWA));
    }

    #[test]
    fn acoustic_profile_distinguishes_high_front_and_back_rounded_vowels() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let high_front = profile
            .phone_models
            .get(&arpabet::phone_id_for_ipa("iː"))
            .expect("IY phone fingerprint");
        let high_back = profile
            .phone_models
            .get(&arpabet::phone_id_for_ipa("uː"))
            .expect("UW phone fingerprint");

        assert_acoustic_category(high_front, "f2_region", "high");
        assert_acoustic_category(high_front, "rounding_resonance", "absent");
        assert_acoustic_category(high_back, "f2_region", "low");
        assert_acoustic_category(high_back, "rounding_resonance", "present");
        assert!(
            high_front
                .landmarks
                .iter()
                .any(|landmark| landmark.kind == AcousticLandmarkKind::VowelTarget)
        );
    }

    #[test]
    fn vowel_fingerprints_include_formant_target_ranges_by_phone() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let iy = profile
            .phone_models
            .get(&arpabet::phone_id_for_ipa("iː"))
            .expect("IY phone fingerprint");
        let uw = profile
            .phone_models
            .get(&arpabet::phone_id_for_ipa("uː"))
            .expect("UW phone fingerprint");
        let er = phoneme_model_by_alias(&ga, profile, "ER");

        assert_range(
            iy,
            AcousticMeasurement::Formant { index: 1 },
            240.0,
            350.0,
            "Hz",
        );
        assert_range(
            iy,
            AcousticMeasurement::Formant { index: 2 },
            2200.0,
            3000.0,
            "Hz",
        );
        assert_range(
            uw,
            AcousticMeasurement::Formant { index: 2 },
            600.0,
            1200.0,
            "Hz",
        );
        assert_range(
            er,
            AcousticMeasurement::Formant { index: 3 },
            1400.0,
            2200.0,
            "Hz",
        );
        assert!(iy.landmarks.iter().any(|landmark| {
            landmark
                .range_targets
                .iter()
                .any(|target| target.measurement == AcousticMeasurement::Formant { index: 1 })
        }));
    }

    #[test]
    fn acoustic_profile_covers_all_inventory_segments() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");

        for phoneme in ga.phonemes.phonemes.values() {
            assert!(
                profile.phoneme_models.contains_key(&phoneme.id),
                "missing acoustic model for phoneme object {:?}",
                phoneme.id
            );
        }

        for phone_id in ga.phones.phones.keys() {
            assert!(
                profile.phone_models.contains_key(phone_id),
                "missing acoustic model for phone object {:?}",
                phone_id
            );
        }
    }

    #[test]
    fn diphthongs_and_r_colored_vowels_carry_extra_vowel_cues() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let ay = phoneme_model_by_alias(&ga, profile, "AY");
        let er = phoneme_model_by_alias(&ga, profile, "ER");
        let r_colored_schwa = profile
            .phone_models
            .get(&R_COLORED_SCHWA)
            .expect("r-colored schwa acoustic model");

        assert_acoustic_category(ay, "formant_trajectory", "low_front_to_high_front");
        assert!(
            ay.landmarks
                .iter()
                .any(|landmark| landmark.kind == AcousticLandmarkKind::FormantTransition)
        );
        assert_acoustic_bool(er, "rhoticity", true);
        assert_acoustic_category(er, "f3_region", "low");
        assert_acoustic_bool(r_colored_schwa, "vowel_reduction", true);
        assert_acoustic_category(r_colored_schwa, "f3_region", "low");
    }

    #[test]
    fn consonant_inventory_segments_carry_manner_specific_acoustic_cues() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let t = phoneme_model_by_alias(&ga, profile, "T");
        let s = phoneme_model_by_alias(&ga, profile, "S");
        let ch = phoneme_model_by_alias(&ga, profile, "CH");
        let m = phoneme_model_by_alias(&ga, profile, "M");
        let l = phoneme_model_by_alias(&ga, profile, "L");
        let tap = profile.phone_models.get(&TAP).expect("tap acoustic model");
        let syllable_break = profile
            .phone_models
            .get(&SYLLABLE_BREAK)
            .expect("syllable break acoustic model");

        assert_acoustic_bool(t, "stop_closure", true);
        assert_acoustic_bool(t, "release_burst", true);
        assert_acoustic_category(t, "stop_burst_spectral_shape", "diffuse_rising");
        assert_acoustic_category(t, "place_formant_locus", "coronal_fronted");
        assert_eq!(
            acoustic_value(t, "aspiration_present"),
            Some(&Spec::Variable(vec![
                FeatureValue::Bool(false),
                FeatureValue::Bool(true)
            ]))
        );
        assert_acoustic_bool(s, "frication_noise", true);
        assert_acoustic_category(s, "frication_spectral_shape", "high_sibilant");
        assert_acoustic_bool(ch, "affricate_release", true);
        assert_acoustic_bool(ch, "frication_noise", true);
        assert_acoustic_bool(m, "nasal_murmur", true);
        assert_acoustic_bool(m, "nasal_antiresonance", true);
        assert_acoustic_category(m, "nasal_place", "labial_murmur");
        assert_acoustic_bool(l, "approximant_formants", true);
        assert_acoustic_bool(l, "lateral_resonance", true);
        assert_acoustic_bool(tap, "tap_closure", true);
        assert_acoustic_category(syllable_break, "segment_boundary", "syllable");
    }

    #[test]
    fn acoustic_profile_marks_bilabial_stop_cues_without_overclaiming_aspiration() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let p = profile.phone_models.get(&P).expect("p phone fingerprint");
        let b = profile.phone_models.get(&B).expect("b phone fingerprint");

        assert_acoustic_bool(p, "stop_closure", true);
        assert_acoustic_bool(p, "release_burst", true);
        assert_acoustic_bool(b, "stop_closure", true);
        assert!(
            p.landmarks
                .iter()
                .any(|landmark| landmark.kind == AcousticLandmarkKind::Aspiration)
        );
        assert_eq!(
            acoustic_value(p, "aspiration_present"),
            Some(&Spec::Variable(vec![
                FeatureValue::Bool(false),
                FeatureValue::Bool(true)
            ]))
        );
        assert_eq!(
            acoustic_value(b, "aspiration_present"),
            Some(&Spec::Known(FeatureValue::Bool(false)))
        );
        assert_acoustic_bool(
            phoneme_model_by_alias(&ga, profile, "P"),
            "stop_closure",
            true,
        );
    }

    #[test]
    fn stop_and_tap_fingerprints_include_vot_and_closure_duration_ranges() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let p = profile.phone_models.get(&P).expect("p phone fingerprint");
        let b = profile.phone_models.get(&B).expect("b phone fingerprint");
        let tap = profile.phone_models.get(&TAP).expect("tap acoustic model");

        assert_range(p, AcousticMeasurement::VoiceOnsetTime, 25.0, 85.0, "ms");
        assert_range(p, AcousticMeasurement::ClosureDuration, 50.0, 130.0, "ms");
        assert_range(b, AcousticMeasurement::VoiceOnsetTime, -80.0, 20.0, "ms");
        assert_range(tap, AcousticMeasurement::ClosureDuration, 10.0, 30.0, "ms");
        assert!(tap.landmarks.iter().any(|landmark| {
            landmark.kind == AcousticLandmarkKind::Closure
                && landmark
                    .range_targets
                    .iter()
                    .any(|target| target.measurement == AcousticMeasurement::ClosureDuration)
        }));
    }

    #[test]
    fn fricative_and_nasal_fingerprints_include_noise_and_resonance_ranges() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let s = profile.phone_models.get(&S).expect("s phone fingerprint");
        let z = profile.phone_models.get(&Z).expect("z phone fingerprint");
        let m = profile.phone_models.get(&M).expect("m phone fingerprint");
        let ng = profile.phone_models.get(&NG).expect("ng phone fingerprint");

        assert_range(s, AcousticMeasurement::FricationDuration, 60.0, 190.0, "ms");
        assert_range(
            s,
            AcousticMeasurement::SpectralCentroid,
            4500.0,
            8500.0,
            "Hz",
        );
        assert_range(z, AcousticMeasurement::FricationDuration, 45.0, 150.0, "ms");
        assert_range(m, AcousticMeasurement::NasalMurmurBand, 200.0, 350.0, "Hz");
        assert_range(
            m,
            AcousticMeasurement::NasalAntiresonance,
            750.0,
            1250.0,
            "Hz",
        );
        assert_range(
            ng,
            AcousticMeasurement::NasalAntiresonance,
            2500.0,
            3500.0,
            "Hz",
        );
    }

    #[test]
    fn fricatives_carry_per_phone_centroid_skew_and_diffuse_strength_cues() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let s = profile.phone_models.get(&S).expect("s phone fingerprint");
        let sh = profile.phone_models.get(&SH).expect("sh phone fingerprint");
        let f = profile.phone_models.get(&F).expect("f phone fingerprint");
        let th = profile.phone_models.get(&TH).expect("th phone fingerprint");
        let h_id = arpabet::phone_id_for_ipa("h");
        let h = profile
            .phone_models
            .get(&h_id)
            .expect("h phone fingerprint");

        assert_acoustic_category(s, "frication_spectral_skew", "strong_high_frequency_skew");
        assert_acoustic_category(s, "frication_strength", "strong_anterior_sibilant");
        assert_range(s, AcousticMeasurement::SpectralSkew, 0.6, 1.4, "unitless");
        assert_acoustic_category(sh, "frication_spectral_skew", "moderate_postalveolar_skew");
        assert_range(sh, AcousticMeasurement::SpectralSkew, -0.2, 0.6, "unitless");
        assert_acoustic_category(f, "frication_strength", "weak_diffuse_labiodental");
        assert_acoustic_category(th, "frication_strength", "weak_diffuse_dental");
        assert_acoustic_category(h, "frication_strength", "weak_diffuse_glottal");
    }

    #[test]
    fn nasals_carry_place_transition_murmur_and_antiresonance_differences() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let m = profile.phone_models.get(&M).expect("m phone fingerprint");
        let n = profile.phone_models.get(&N).expect("n phone fingerprint");
        let ng = profile.phone_models.get(&NG).expect("ng phone fingerprint");

        assert_acoustic_category(m, "nasal_place_transition", "low_f2_labial_transition");
        assert_acoustic_category(n, "nasal_place_transition", "fronted_coronal_transition");
        assert_acoustic_category(ng, "nasal_place_transition", "velar_pinch_transition");
        assert_range(
            m,
            AcousticMeasurement::NasalPlaceTransition,
            800.0,
            1300.0,
            "Hz",
        );
        assert_range(
            n,
            AcousticMeasurement::NasalPlaceTransition,
            1600.0,
            2300.0,
            "Hz",
        );
        assert_range(
            ng,
            AcousticMeasurement::NasalPlaceTransition,
            1100.0,
            2600.0,
            "Hz",
        );
    }

    #[test]
    fn approximants_carry_specific_formant_transition_patterns() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let w = phoneme_model_by_alias(&ga, profile, "W");
        let y = phoneme_model_by_alias(&ga, profile, "Y");
        let r = phoneme_model_by_alias(&ga, profile, "R");
        let l = phoneme_model_by_alias(&ga, profile, "L");
        let dark_l = profile
            .phone_models
            .get(&DARK_L)
            .expect("dark l phone fingerprint");

        assert_acoustic_category(
            w,
            "approximant_formant_transition_detail",
            "low_f2_rounded_glide",
        );
        assert_range(
            w,
            AcousticMeasurement::FormantTransition { index: 2 },
            -900.0,
            -250.0,
            "Hz_delta",
        );
        assert_acoustic_category(
            y,
            "approximant_formant_transition_detail",
            "high_f2_palatal_glide",
        );
        assert_range(
            y,
            AcousticMeasurement::FormantTransition { index: 2 },
            500.0,
            1400.0,
            "Hz_delta",
        );
        assert_acoustic_category(
            r,
            "approximant_formant_transition_detail",
            "lowered_f3_rhotic_transition",
        );
        assert_range(
            r,
            AcousticMeasurement::FormantTransition { index: 3 },
            -1200.0,
            -350.0,
            "Hz_delta",
        );
        assert_acoustic_category(
            l,
            "approximant_formant_transition_detail",
            "coronal_lateral_f2_transition",
        );
        assert_acoustic_category(
            dark_l,
            "approximant_formant_transition_detail",
            "dark_l_low_f2_velarized_transition",
        );
        assert_range(
            dark_l,
            AcousticMeasurement::FormantTransition { index: 2 },
            -700.0,
            -100.0,
            "Hz_delta",
        );
    }

    #[test]
    fn affricates_carry_closure_to_frication_timing() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let ch = phoneme_model_by_alias(&ga, profile, "CH");
        let jh = phoneme_model_by_alias(&ga, profile, "JH");

        assert_acoustic_category(
            ch,
            "affricate_closure_to_frication_timing",
            "short_voiceless_affricate_lag",
        );
        assert_range(
            ch,
            AcousticMeasurement::AffricateClosureToFrication,
            8.0,
            35.0,
            "ms",
        );
        assert_acoustic_category(
            jh,
            "affricate_closure_to_frication_timing",
            "short_voiced_affricate_lag",
        );
        assert_range(
            jh,
            AcousticMeasurement::AffricateClosureToFrication,
            5.0,
            30.0,
            "ms",
        );
    }

    #[test]
    fn stops_carry_closure_burst_aspiration_voicing_temporal_order() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let aspirated_p = profile
            .phone_models
            .get(&ASPIRATED_P)
            .expect("aspirated p acoustic model");

        assert_eq!(
            temporal_order(aspirated_p),
            vec![
                AcousticLandmarkKind::Closure,
                AcousticLandmarkKind::ReleaseBurst,
                AcousticLandmarkKind::Aspiration,
                AcousticLandmarkKind::VoicingOnset,
            ]
        );
        assert!(
            aspirated_p
                .landmarks
                .iter()
                .any(|landmark| landmark.kind == AcousticLandmarkKind::VoicingOnset)
        );
        assert_eq!(
            aspirated_p.temporal.sampling_strategy,
            Some(SegmentSamplingStrategy::UseOffsetTransition)
        );
    }

    #[test]
    fn temporal_models_capture_subsegment_proportions() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let aspirated_p = profile
            .phone_models
            .get(&ASPIRATED_P)
            .expect("aspirated p acoustic model");
        let tap = profile.phone_models.get(&TAP).expect("tap acoustic model");
        let ch = phoneme_model_by_alias(&ga, profile, "CH");
        let ay = phoneme_model_by_alias(&ga, profile, "AY");

        assert_subsegment(aspirated_p, SubsegmentRole::Aspiration, 0.15, 0.4);
        assert_subsegment(tap, SubsegmentRole::TapClosure, 0.55, 0.95);
        assert_subsegment(ch, SubsegmentRole::Closure, 0.35, 0.55);
        assert_subsegment(ch, SubsegmentRole::Frication, 0.35, 0.6);
        assert_subsegment(ay, SubsegmentRole::VowelOnsetTransition, 0.25, 0.4);
        assert_subsegment(ay, SubsegmentRole::VowelOffsetTransition, 0.35, 0.55);
    }

    #[test]
    fn temporal_models_mark_midpoint_vs_transition_sampling() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let iy = profile
            .phone_models
            .get(&arpabet::phone_id_for_ipa("iː"))
            .expect("IY phone fingerprint");
        let ay = phoneme_model_by_alias(&ga, profile, "AY");
        let s = profile.phone_models.get(&S).expect("s phone fingerprint");
        let n = profile.phone_models.get(&N).expect("n phone fingerprint");
        let w = phoneme_model_by_alias(&ga, profile, "W");

        assert_eq!(
            iy.temporal.sampling_strategy,
            Some(SegmentSamplingStrategy::UseMidpoint)
        );
        assert_eq!(
            ay.temporal.sampling_strategy,
            Some(SegmentSamplingStrategy::UseFullTrajectory)
        );
        assert_eq!(
            s.temporal.sampling_strategy,
            Some(SegmentSamplingStrategy::UseMidpoint)
        );
        assert_eq!(
            n.temporal.sampling_strategy,
            Some(SegmentSamplingStrategy::UseOnsetAndOffsetTransitions)
        );
        assert_eq!(
            w.temporal.sampling_strategy,
            Some(SegmentSamplingStrategy::UseFullTrajectory)
        );
    }

    #[test]
    fn acoustic_cues_mark_diagnosticity_and_dependencies() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let vowel_nucleus = cue_def(profile, "acoustic.cue.vowel_nucleus");
        let nasal_place = cue_def(profile, "acoustic.cue.nasal_place");
        let f1 = cue_def(profile, "acoustic.cue.f1_region");
        let aspiration = cue_def(profile, "acoustic.cue.aspiration_noise");
        let boundary_gap = cue_def(profile, "acoustic.cue.boundary_gap");

        assert_eq!(vowel_nucleus.diagnosticity, CueDiagnosticity::Robust);
        assert_eq!(nasal_place.diagnosticity, CueDiagnosticity::Weak);
        assert!(f1.dependencies.contains(&CueDependency::SpeakerDependent));
        assert!(f1.dependencies.contains(&CueDependency::ContextDependent));
        assert!(
            aspiration
                .dependencies
                .contains(&CueDependency::StyleDependent)
        );
        assert_eq!(boundary_gap.diagnosticity, CueDiagnosticity::Weak);
        assert!(
            boundary_gap
                .dependencies
                .contains(&CueDependency::ContextDependent)
        );
    }

    #[test]
    fn boundary_and_silence_models_distinguish_alignment_points_from_pauses() {
        let ga = variety("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let word = profile
            .phone_models
            .get(&WORD_BOUNDARY)
            .expect("word boundary acoustic model");
        let letter = profile
            .phone_models
            .get(&LETTER_BOUNDARY)
            .expect("letter boundary acoustic model");
        let phrase = profile
            .phone_models
            .get(&PHRASE_PAUSE)
            .expect("phrase pause acoustic model");
        let terminal = profile
            .phone_models
            .get(&TERMINAL_PAUSE)
            .expect("terminal pause acoustic model");

        assert_acoustic_category(word, "segment_boundary", "word");
        assert_acoustic_category(word, "boundary_gap", "none");
        assert_acoustic_bool(word, "silent_boundary", false);
        assert_range(word, AcousticMeasurement::SilenceDuration, 0.0, 20.0, "ms");
        assert_acoustic_category(letter, "segment_boundary", "letter");
        assert_acoustic_bool(letter, "silent_boundary", false);
        assert_acoustic_category(phrase, "boundary_gap", "phrase_pause");
        assert_acoustic_bool(phrase, "silent_boundary", true);
        assert_range(
            phrase,
            AcousticMeasurement::SilenceDuration,
            120.0,
            450.0,
            "ms",
        );
        assert_acoustic_category(terminal, "boundary_gap", "terminal_pause");
        assert_acoustic_bool(terminal, "silent_boundary", true);
        assert_range(
            terminal,
            AcousticMeasurement::SilenceDuration,
            350.0,
            1200.0,
            "ms",
        );
    }

    #[test]
    fn rules_are_variety_data() {
        let ga = variety("en-US-GA");
        let flapping = ga
            .allophone_rules
            .iter()
            .find(|rule| rule.id == "american_english_intervocalic_flapping")
            .expect("flapping rule");

        assert_eq!(flapping.status, RuleStatus::StyleDependent);
        assert!(
            flapping
                .conditions
                .contains(&RuleCondition::NotCarefulStyle)
        );
        assert_eq!(flapping.environment.word_position, Spec::Unspecified);
        assert_eq!(flapping.environment.prosodic_context, Spec::Unspecified);
    }

    #[test]
    fn phonemes_contain_allophones_with_environments() {
        let ga = variety("en-US-GA");
        let t = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "T"))
            .expect("T phoneme");
        let tap = t
            .allophones
            .iter()
            .find(|allophone| allophone.phone == TAP)
            .expect("tap allophone");

        assert!(t.possible_phones.contains(&TAP));
        assert_eq!(
            tap.source_rule_id.as_deref(),
            Some("american_english_intervocalic_flapping")
        );
        assert_eq!(tap.status, RuleStatus::StyleDependent);
        assert_eq!(tap.environment.before.len(), 1);
        assert_eq!(tap.environment.after.len(), 1);
        assert!(tap.conditions.contains(&RuleCondition::NotCarefulStyle));

        let n = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "N"))
            .expect("N phoneme");
        assert!(
            n.allophones
                .iter()
                .any(|allophone| allophone.phone == NG && allophone.environment.after.len() == 1)
        );
    }

    #[test]
    fn context_conditioned_allophones_are_inventory_data() {
        let ga = variety("en-US-GA");
        let p = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "P"))
            .expect("P phoneme");
        let d = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "D"))
            .expect("D phoneme");
        let l = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "L"))
            .expect("L phoneme");
        let ah = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "AH"))
            .expect("AH phoneme");

        assert!(p.possible_phones.contains(&ASPIRATED_P));
        assert!(p.possible_phones.contains(&UNASPIRATED_P));
        assert!(d.possible_phones.contains(&TAP));
        assert!(l.possible_phones.contains(&DARK_L));
        assert!(ah.possible_phones.contains(&SCHWA));
        assert!(ah.possible_phones.contains(&STRUT));
        assert!(ah.allophones.iter().any(|allophone| {
            allophone.phone == SCHWA
                && allophone.environment.stress_context == Spec::Known(Stress::Unstressed)
                && allophone.environment.syllable_position == Spec::Known(SyllablePosition::Nucleus)
        }));
        assert!(ah.allophones.iter().any(|allophone| {
            allophone.phone == STRUT
                && allophone.environment.stress_context == Spec::Known(Stress::Primary)
                && allophone.environment.syllable_position == Spec::Known(SyllablePosition::Nucleus)
        }));
        assert!(ah.allophones.iter().any(|allophone| {
            allophone.phone == STRUT
                && allophone.environment.stress_context == Spec::Known(Stress::Secondary)
                && allophone.environment.syllable_position == Spec::Known(SyllablePosition::Nucleus)
        }));

        let er = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "ER"))
            .expect("ER phoneme");
        assert!(er.possible_phones.contains(&R_COLORED_SCHWA));
        assert!(er.possible_phones.contains(&STRESSED_RHOTIC_VOWEL));
        assert!(er.allophones.iter().any(|allophone| {
            allophone.phone == R_COLORED_SCHWA
                && allophone.environment.stress_context == Spec::Known(Stress::Unstressed)
                && allophone.environment.syllable_position == Spec::Known(SyllablePosition::Nucleus)
        }));
        assert!(er.allophones.iter().any(|allophone| {
            allophone.phone == STRESSED_RHOTIC_VOWEL
                && allophone.environment.stress_context == Spec::Known(Stress::Primary)
                && allophone.environment.syllable_position == Spec::Known(SyllablePosition::Nucleus)
        }));
        assert!(er.allophones.iter().any(|allophone| {
            allophone.phone == STRESSED_RHOTIC_VOWEL
                && allophone.environment.stress_context == Spec::Known(Stress::Secondary)
                && allophone.environment.syllable_position == Spec::Known(SyllablePosition::Nucleus)
        }));

        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let aspirated_p = profile
            .phone_models
            .get(&ASPIRATED_P)
            .expect("aspirated p acoustic model");
        let unaspirated_p = profile
            .phone_models
            .get(&UNASPIRATED_P)
            .expect("unaspirated p acoustic model");
        let dark_l = profile
            .phone_models
            .get(&DARK_L)
            .expect("dark l acoustic model");

        assert_acoustic_bool(aspirated_p, "aspiration_present", true);
        assert_acoustic_bool(unaspirated_p, "aspiration_present", false);
        assert_acoustic_category(dark_l, "l_quality", "dark");
    }

    #[test]
    fn weak_forms_are_variety_data() {
        let ga = variety("en-US-GA");
        let weak_the = ga
            .weak_forms
            .iter()
            .find(|rule| rule.id == "english_weak_the_before_consonant")
            .expect("weak form for the before consonants");

        assert_eq!(weak_the.lexical_item, "the");
        assert_eq!(
            weak_the.pronunciation,
            vec![
                arpabet::phoneme_id("en-US-GA", "DH"),
                arpabet::phoneme_id("en-US-GA", "AH0")
            ]
        );
        assert_eq!(
            weak_the.following,
            WeakFormFollowingContext::BeforeConsonantish
        );
    }

    fn assert_acoustic_category(model: &AcousticTargetModel, name: &str, expected: &str) {
        assert_eq!(
            acoustic_value(model, name),
            Some(&Spec::Known(FeatureValue::Category(expected.into())))
        );
    }

    fn assert_acoustic_bool(model: &AcousticTargetModel, name: &str, expected: bool) {
        assert_eq!(
            acoustic_value(model, name),
            Some(&Spec::Known(FeatureValue::Bool(expected)))
        );
    }

    fn assert_range(
        model: &AcousticTargetModel,
        measurement: AcousticMeasurement,
        min: f32,
        max: f32,
        unit: &str,
    ) {
        let target = model
            .range_targets
            .iter()
            .find(|target| target.measurement == measurement)
            .expect("acoustic range target");
        assert_eq!(target.range.min, min);
        assert_eq!(target.range.max, max);
        assert_eq!(target.range.unit, unit);
    }

    fn temporal_order(model: &AcousticTargetModel) -> Vec<AcousticLandmarkKind> {
        model
            .temporal
            .landmark_order
            .iter()
            .map(|step| step.kind.clone())
            .collect()
    }

    fn assert_subsegment(model: &AcousticTargetModel, role: SubsegmentRole, min: f32, max: f32) {
        let subsegment = model
            .temporal
            .subsegments
            .iter()
            .find(|subsegment| subsegment.role == role)
            .expect("temporal subsegment");
        assert_eq!(subsegment.proportion.min, min);
        assert_eq!(subsegment.proportion.max, max);
        assert_eq!(subsegment.proportion.unit, "proportion");
    }

    fn cue_def<'a>(profile: &'a AcousticProfile, id: &str) -> &'a AcousticCueDef {
        profile
            .cues
            .get(&AcousticCueId(id.into()))
            .expect("acoustic cue definition")
    }

    fn acoustic_value<'a>(
        model: &'a AcousticTargetModel,
        name: &str,
    ) -> Option<&'a Spec<FeatureValue>> {
        model
            .expected_features
            .values
            .get(&FeatureId(format!("acoustic.{name}")))
    }

    fn phoneme_model_by_alias<'a>(
        variety: &'a LinguisticVariety,
        profile: &'a AcousticProfile,
        alias: &str,
    ) -> &'a AcousticTargetModel {
        let phoneme = variety
            .phonemes
            .phonemes
            .values()
            .find(|phoneme| {
                phoneme
                    .aliases
                    .iter()
                    .any(|candidate| candidate.system == "arpabet" && candidate.symbol == alias)
            })
            .expect("phoneme alias");
        profile
            .phoneme_models
            .get(&phoneme.id)
            .expect("phoneme acoustic model")
    }
}
