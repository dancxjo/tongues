pub mod lexicons;
pub mod notation;
pub mod varieties;

pub use lexicons::cmudict;
pub use notation::arpabet;
pub use varieties::{builtin_varieties, canonical_variety_id, variety_by_code};
pub use varieties::{english, esperanto, spanish};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::VarietyId;
    use crate::variety::VarietyImplementationStatus;

    #[test]
    fn codes_select_varieties_without_variety_specific_api() {
        assert_eq!(canonical_variety_id("en-US").unwrap().0, "en-US-GA");
        assert_eq!(canonical_variety_id("en-US-GA").unwrap().0, "en-US-GA");
        assert!(variety_by_code("en-US").is_some());
        assert!(variety_by_code("eo").is_some());
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
}
