pub mod english;
pub mod esperanto;

use crate::ids::VarietyId;
use crate::variety::LinguisticVariety;

pub fn canonical_variety_id(code: &str) -> Option<VarietyId> {
    let id = match code {
        "en-US" => "en-US-GA",
        "en-US-GA" | "en-US-singing" | "en-GB-RP" | "en-GB-ScotE" | "en-US-AAE" => code,
        "eo" => "eo",
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
    ]
    .into_iter()
    .filter_map(variety_by_code)
    .collect()
}
