//! Native MBROLA voice-database synthesis.
//!
//! The canonical input remains [`speaking::UtterancePlan`]. This module lowers
//! it to an inspectable timed-phone plan, supports standards-compatible `.pho`
//! interchange, and renders MBROLA diphones in Rust without invoking the
//! historical `mbrola` executable.

mod database;
mod pho;
mod projector;
mod render;

pub use database::{MbrolaDatabase, MbrolaDatabaseError, MbrolaDiphone};
pub use pho::{
    parse_pho, serialize_pho, MbrolaPhoError, MbrolaPhone, MbrolaPitchTarget, PhoneTimedPlan,
};
pub use projector::{
    MbrolaLoweringError, MbrolaLoweringReport, MbrolaProjector, MbrolaSymbolMap,
    MbrolaTimingProfile, MbrolaVoiceMetadata, MBROLA_SILENCE,
};
pub use render::NativeMbrolaRenderer;
