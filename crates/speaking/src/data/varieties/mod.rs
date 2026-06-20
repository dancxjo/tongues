pub mod english;
pub mod esperanto;
pub mod french;
pub mod german;
pub mod greek;
pub mod latin;
pub mod sanskrit;
pub mod spanish;

use crate::ids::VarietyId;
use crate::variety::LinguisticVariety;

struct VarietyRegistration {
    canonical_id: &'static str,
    aliases: &'static [&'static str],
    load: fn(&str) -> LinguisticVariety,
}

const BUILTIN_VARIETY_REGISTRY: &[VarietyRegistration] = &[
    VarietyRegistration {
        canonical_id: "en-US-GA",
        aliases: &["en-US"],
        load: english_variety,
    },
    VarietyRegistration {
        canonical_id: "en-US-singing",
        aliases: &[],
        load: english_variety,
    },
    VarietyRegistration {
        canonical_id: "en-GB-RP",
        aliases: &[],
        load: english_variety,
    },
    VarietyRegistration {
        canonical_id: "en-GB-ScotE",
        aliases: &[],
        load: english_variety,
    },
    VarietyRegistration {
        canonical_id: "en-US-AAE",
        aliases: &[],
        load: english_variety,
    },
    VarietyRegistration {
        canonical_id: "eo",
        aliases: &[],
        load: esperanto_variety,
    },
    VarietyRegistration {
        canonical_id: "fr-FR-Standard",
        aliases: &["fr", "fra", "fr-FR"],
        load: french_variety,
    },
    VarietyRegistration {
        canonical_id: "de-DE-Standard",
        aliases: &["de", "deu", "de-DE"],
        load: german_variety,
    },
    VarietyRegistration {
        canonical_id: "el-GR-Standard",
        aliases: &["el", "el-GR"],
        load: greek_variety,
    },
    VarietyRegistration {
        canonical_id: "grc-Attic",
        aliases: &["grc", "grc-Ancient"],
        load: greek_variety,
    },
    VarietyRegistration {
        canonical_id: "grc-Koine",
        aliases: &["el-Koine"],
        load: greek_variety,
    },
    VarietyRegistration {
        canonical_id: "la-Classical",
        aliases: &["la"],
        load: latin_variety,
    },
    VarietyRegistration {
        canonical_id: "la-Ecclesiastical",
        aliases: &["la-Church"],
        load: latin_variety,
    },
    VarietyRegistration {
        canonical_id: "sa-Deva-Standard",
        aliases: &["sa", "san", "sa-Deva"],
        load: sanskrit_variety,
    },
    VarietyRegistration {
        canonical_id: "es-ES-Castilian",
        aliases: &["es", "es-ES"],
        load: spanish_variety,
    },
    VarietyRegistration {
        canonical_id: "es-419-Standard",
        aliases: &["es-419", "es-LatAm"],
        load: spanish_variety,
    },
];

pub fn canonical_variety_id(code: &str) -> Option<VarietyId> {
    find_variety_registration(code).map(|registration| VarietyId(registration.canonical_id.into()))
}

pub fn variety_by_code(code: &str) -> Option<LinguisticVariety> {
    let canonical = canonical_variety_id(code)?;
    let registration = find_variety_registration(&canonical.0)?;
    Some((registration.load)(registration.canonical_id))
}

pub fn builtin_varieties() -> Vec<LinguisticVariety> {
    BUILTIN_VARIETY_REGISTRY
        .iter()
        .map(|registration| (registration.load)(registration.canonical_id))
        .collect()
}

fn find_variety_registration(code: &str) -> Option<&'static VarietyRegistration> {
    BUILTIN_VARIETY_REGISTRY.iter().find(|registration| {
        registration.canonical_id == code || registration.aliases.contains(&code)
    })
}

fn english_variety(id: &str) -> LinguisticVariety {
    english::variety(id)
}

fn esperanto_variety(_id: &str) -> LinguisticVariety {
    esperanto::variety()
}

fn french_variety(_id: &str) -> LinguisticVariety {
    french::variety()
}

fn german_variety(_id: &str) -> LinguisticVariety {
    german::variety()
}

fn greek_variety(id: &str) -> LinguisticVariety {
    greek::variety(id)
}

fn latin_variety(id: &str) -> LinguisticVariety {
    latin::variety(id)
}

fn sanskrit_variety(_id: &str) -> LinguisticVariety {
    sanskrit::variety()
}

fn spanish_variety(id: &str) -> LinguisticVariety {
    spanish::variety(id)
}
