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

pub fn canonical_variety_id(code: &str) -> Option<VarietyId> {
    let id = match code {
        "en-US" => "en-US-GA",
        "en-US-GA" | "en-US-singing" | "en-GB-RP" | "en-GB-ScotE" | "en-US-AAE" => code,
        "eo" => "eo",
        "fr" | "fra" | "fr-FR" | "fr-FR-Standard" => "fr-FR-Standard",
        "de" | "deu" | "de-DE" | "de-DE-Standard" => "de-DE-Standard",
        "el" | "el-GR" | "el-GR-Standard" => "el-GR-Standard",
        "grc" | "grc-Attic" | "grc-Ancient" => "grc-Attic",
        "grc-Koine" | "el-Koine" => "grc-Koine",
        "la" | "la-Classical" => "la-Classical",
        "la-Ecclesiastical" | "la-Church" => "la-Ecclesiastical",
        "sa" | "san" | "sa-Deva" | "sa-Deva-Standard" => "sa-Deva-Standard",
        "es" | "es-ES" => "es-ES-Castilian",
        "es-ES-Castilian" => code,
        "es-419" | "es-LatAm" => "es-419-Standard",
        "es-419-Standard" => code,
        _ => return None,
    };
    Some(VarietyId(id.to_string()))
}

pub fn variety_by_code(code: &str) -> Option<LinguisticVariety> {
    let canonical = canonical_variety_id(code)?;
    match canonical.0.as_str() {
        "en-US-GA" => Some(english::variety("en-US-GA")),
        "en-US-singing" => Some(english::variety("en-US-singing")),
        "en-GB-RP" => Some(english::variety("en-GB-RP")),
        "en-GB-ScotE" => Some(english::variety("en-GB-ScotE")),
        "en-US-AAE" => Some(english::variety("en-US-AAE")),
        "eo" => Some(esperanto::variety()),
        "fr-FR-Standard" => Some(french::variety()),
        "de-DE-Standard" => Some(german::variety()),
        "el-GR-Standard" => Some(greek::variety("el-GR-Standard")),
        "grc-Attic" => Some(greek::variety("grc-Attic")),
        "grc-Koine" => Some(greek::variety("grc-Koine")),
        "la-Classical" => Some(latin::variety("la-Classical")),
        "la-Ecclesiastical" => Some(latin::variety("la-Ecclesiastical")),
        "sa-Deva-Standard" => Some(sanskrit::variety()),
        "es-ES-Castilian" => Some(spanish::variety("es-ES-Castilian")),
        "es-419-Standard" => Some(spanish::variety("es-419-Standard")),
        _ => None,
    }
}

pub fn builtin_varieties() -> Vec<LinguisticVariety> {
    [
        "en-US-GA",
        "en-US-singing",
        "en-GB-RP",
        "en-GB-ScotE",
        "en-US-AAE",
        "eo",
        "fr-FR-Standard",
        "de-DE-Standard",
        "el-GR-Standard",
        "grc-Attic",
        "grc-Koine",
        "la-Classical",
        "la-Ecclesiastical",
        "sa-Deva-Standard",
        "es-ES-Castilian",
        "es-419-Standard",
    ]
    .into_iter()
    .filter_map(variety_by_code)
    .collect()
}
