pub mod cmudict;
pub mod lexique;

pub const CMUDICT_ID: &str = "cmudict";
pub const LEXIQUE383_ID: &str = "lexique383";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexiconAdapter {
    ArpabetDictionary,
    IpaDictionary,
}

pub const LEXICON_IDS: &[&str] = &[CMUDICT_ID, LEXIQUE383_ID];

pub fn adapter_for_id(id: &str) -> Option<LexiconAdapter> {
    match id {
        CMUDICT_ID => Some(LexiconAdapter::ArpabetDictionary),
        LEXIQUE383_ID => Some(LexiconAdapter::IpaDictionary),
        _ => None,
    }
}
