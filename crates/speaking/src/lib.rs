//! Backend-free linguistic and acoustic speech ontology for mortar-sea.
//!
//! This crate defines what speech is inside the system. ASR, TTS, vocoders,
//! aligners, and neural models should adapt to or from these types rather than
//! leaking backend-specific concepts into the core ontology.

pub mod acoustics;
pub mod asr;
pub mod data;
pub mod evidence;
pub mod feature;
pub mod ids;
pub mod morphology;
pub mod orthography;
pub mod phonemicize;
pub mod phonetics;
pub mod phonology;
pub mod prosody;
pub mod realize;
pub mod rules;
pub mod segment;
pub mod spec;
pub mod streaming;
pub mod syllabify;
pub mod syntax;
pub mod text_stability;
pub mod time;
pub mod transcript;
pub mod utterance;
pub mod variety;
#[cfg(feature = "asr-whisper")]
pub mod whisper;
pub mod word_stream;

pub use acoustics::*;
pub use asr::*;
pub use data::*;
pub use evidence::*;
pub use feature::*;
pub use ids::*;
pub use morphology::*;
pub use orthography::*;
pub use phonemicize::*;
pub use phonetics::*;
pub use phonology::*;
pub use prosody::*;
pub use realize::*;
pub use rules::*;
pub use segment::*;
pub use spec::*;
pub use streaming::*;
pub use syllabify::*;
pub use syntax::*;
pub use text_stability::*;
pub use time::*;
pub use transcript::*;
pub use utterance::*;
pub use variety::*;
#[cfg(feature = "asr-whisper")]
pub use whisper::*;
pub use word_stream::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_and_unspecified_are_distinct() {
        let unknown: Spec<bool> = Spec::Unknown;
        let unspecified: Spec<bool> = Spec::Unspecified;

        assert_ne!(unknown, unspecified);
    }

    #[test]
    fn phone_and_phoneme_are_separate_categories() {
        let t_phoneme = PhonemeId("en-US.phoneme.t".into());
        let tap_phone = PhoneId::from("ipa.phone.tap");

        assert_ne!(t_phoneme.0, tap_phone.as_str());
    }

    #[test]
    fn timespan_duration_never_negative() {
        let span = TimeSpan {
            start_s: 2.0,
            end_s: 1.0,
        };

        assert_eq!(span.duration_s(), 0.0);
    }
}
