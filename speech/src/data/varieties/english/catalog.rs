#[derive(Debug, Clone, Copy)]
pub(super) struct EnglishVarietyRow {
    pub(super) id: &'static str,
    pub(super) name: &'static str,
    pub(super) implementation_status: ImplementationStatusSpec,
    pub(super) singing: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ImplementationStatusSpec {
    Complete,
    StubDerivedFrom(&'static str),
    PermissiveProfile,
}

pub(super) const VARIETIES: &[EnglishVarietyRow] = &[
    EnglishVarietyRow {
        id: "en-US-GA",
        name: "General American English",
        implementation_status: ImplementationStatusSpec::Complete,
        singing: false,
    },
    EnglishVarietyRow {
        id: "en-US-singing",
        name: "Permissive Singing Profile",
        implementation_status: ImplementationStatusSpec::PermissiveProfile,
        singing: true,
    },
    EnglishVarietyRow {
        id: "en-GB-RP",
        name: "Received Pronunciation (stub)",
        implementation_status: ImplementationStatusSpec::StubDerivedFrom("en-US-GA"),
        singing: false,
    },
    EnglishVarietyRow {
        id: "en-GB-ScotE",
        name: "Scottish English (stub)",
        implementation_status: ImplementationStatusSpec::StubDerivedFrom("en-US-GA"),
        singing: false,
    },
    EnglishVarietyRow {
        id: "en-US-AAE",
        name: "African American English (stub)",
        implementation_status: ImplementationStatusSpec::StubDerivedFrom("en-US-GA"),
        singing: false,
    },
];

pub(super) fn get(id: &str) -> &'static EnglishVarietyRow {
    VARIETIES
        .iter()
        .find(|row| row.id == id)
        .unwrap_or(&VARIETIES[0])
}
