use serde::{Deserialize, Serialize};

use crate::feature::FeatureBundle;
use crate::ids::PhoneId;
use crate::segment::{SegmentStatus, SymbolAlias};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phone {
    pub id: PhoneId,
    pub ipa: String,
    pub features: FeatureBundle,
    pub aliases: Vec<SymbolAlias>,
    pub status: SegmentStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PhoneInventory {
    pub phones: std::collections::HashMap<PhoneId, Phone>,
}
