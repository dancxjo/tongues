pub mod lexicons;
pub mod notation;
pub mod varieties;

pub use lexicons::cmudict;
pub use notation::arpabet;
pub use varieties::{
    builtin_languages, builtin_varieties, canonical_variety_id, language_tag_for_variety,
    variety_by_code, wiktionary_language_for_variety,
};
pub use varieties::{english, esperanto, french, german, greek, latin, sanskrit, spanish};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::VarietyId;
    use crate::variety::VarietyImplementationStatus;

    #[test]
    fn codes_select_varieties_without_variety_specific_api() {
        assert_eq!(canonical_variety_id("en-US").unwrap().0, "en-US-GA");
        assert_eq!(canonical_variety_id("en-US.GenAm").unwrap().0, "en-US-GA");
        assert_eq!(canonical_variety_id("EN-us").unwrap().0, "en-US-GA");
        assert_eq!(canonical_variety_id("eng").unwrap().0, "en-US-GA");
        assert_eq!(canonical_variety_id("en-US-GA").unwrap().0, "en-US-GA");
        assert!(variety_by_code("en-US").is_some());
        assert!(variety_by_code("eo").is_some());
        assert_eq!(canonical_variety_id("fra").unwrap().0, "fr-FR-Standard");
        assert_eq!(canonical_variety_id("Fr-fr").unwrap().0, "fr-FR-Standard");
        assert_eq!(canonical_variety_id("deu").unwrap().0, "de-DE-Standard");
        assert_eq!(canonical_variety_id("el").unwrap().0, "el-GR-Standard");
        assert_eq!(canonical_variety_id("grc").unwrap().0, "grc-Attic");
        assert_eq!(canonical_variety_id("grc-Koine").unwrap().0, "grc-Koine");
        assert_eq!(canonical_variety_id("la").unwrap().0, "la-Classical");
        assert_eq!(
            canonical_variety_id("la-Ecclesiastical").unwrap().0,
            "la-Ecclesiastical"
        );
        assert_eq!(canonical_variety_id("san").unwrap().0, "sa-Deva-Standard");
        assert_eq!(canonical_variety_id("es").unwrap().0, "es-ES-Castilian");
        assert_eq!(canonical_variety_id("es-419").unwrap().0, "es-419-Standard");
        assert!(variety_by_code("es-ES-Castilian").is_some());
    }

    #[test]
    fn english_stub_status_is_explicit_data() {
        for code in ["en-GB-RP", "en-GB-ScotE", "en-US-AAE"] {
            let variety = variety_by_code(code).expect("variety");
            assert_eq!(
                variety.implementation_status,
                VarietyImplementationStatus::StubDerivedFrom(VarietyId("en-US-GA".into()))
            );
        }
    }

    #[test]
    fn builtin_varieties_advertise_pipeline_capabilities_as_data() {
        for variety in builtin_varieties() {
            assert!(
                variety
                    .orthography
                    .as_ref()
                    .and_then(|orthography| orthography.pronunciation.as_ref())
                    .is_some(),
                "{} should declare an orthographic pronunciation profile",
                variety.id.0
            );
            assert!(
                variety.orthography.as_ref().is_some_and(|orthography| {
                    !orthography.sample_words.is_empty()
                        && orthography.sample_letter_units.len() >= 2
                }),
                "{} should declare orthography sample words and letters",
                variety.id.0
            );
            assert!(
                variety.syntax_profile.is_some(),
                "{} should declare its syntax profile",
                variety.id.0
            );
            assert!(
                variety.pronunciation_pipeline.is_some(),
                "{} should declare its pronunciation pipeline",
                variety.id.0
            );
            assert!(
                !matches!(
                    variety.text_normalization.number_normalization,
                    crate::variety::NumberNormalizationProfile::None
                ),
                "{} should declare its text normalization profile",
                variety.id.0
            );
            assert!(
                variety
                    .number_names
                    .as_ref()
                    .is_some_and(|names| names.cardinal_0_to_20.len() == 21),
                "{} should declare small-number names",
                variety.id.0
            );
            assert!(
                variety.prosody_profile.as_ref().is_some_and(|profile| {
                    profile.rhythm_class.is_some()
                        && profile.default_rate_syllables_per_second.is_some()
                }),
                "{} should declare a prosody profile",
                variety.id.0
            );
            assert!(
                language_tag_for_variety(&variety.id.0).is_some(),
                "{} should declare its external language tag in variety data",
                variety.id.0
            );
        }
    }

    #[test]
    fn published_language_and_fallback_metadata_are_registry_derived() {
        let registered_varieties = builtin_varieties();
        let registered_language_ids = registered_varieties
            .iter()
            .map(|variety| variety.language.0.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let published_languages = builtin_languages();
        let published_language_ids = published_languages
            .iter()
            .map(|language| language.id.0.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(published_language_ids, registered_language_ids);

        for variety in registered_varieties {
            assert!(
                language_tag_for_variety(&variety.id.0).is_some(),
                "{} should have a registration-owned language tag",
                variety.id.0
            );
            assert!(
                wiktionary_language_for_variety(&variety.id.0).is_some(),
                "{} should derive a Wiktionary language from its registered language",
                variety.id.0
            );
        }

        assert_eq!(
            wiktionary_language_for_variety("fr-FR-Standard"),
            Some("fra")
        );
        assert_eq!(wiktionary_language_for_variety("not-a-variety"), None);
    }
}
