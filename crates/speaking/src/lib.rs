//! Backend-free linguistic and acoustic speech ontology for mortar-sea.
//!
//! This crate defines what speech is inside the system. ASR, TTS, vocoders,
//! aligners, and neural models should adapt to or from these types rather than
//! leaking backend-specific concepts into the core ontology.

pub mod acoustics;
pub mod asr;
pub mod conformance;
pub mod data;
pub mod discrepancies;
pub mod duplex;
pub mod event;
pub mod evidence;
pub mod feature;
pub mod ids;
pub mod incremental_morphology;
pub mod morphology;
pub mod orthography;
pub mod phonemicize;
pub mod phonetics;
pub mod phonology;
pub mod plan_projection;
pub mod prosody;
pub mod realize;
pub mod repair_delivery;
pub mod rules;
pub mod segment;
pub mod spec;
pub mod streaming;
pub mod syllabify;
pub mod syntax;
pub mod text_stability;
pub mod time;
pub mod transcript;
pub mod tts_ledger;
pub mod utterance;
pub mod variety;
#[cfg(feature = "asr-whisper")]
pub mod whisper;
pub mod word_stream;

pub use acoustics::*;
pub use asr::*;
pub use conformance::*;
pub use data::*;
pub use discrepancies::*;
pub use duplex::*;
pub use event::*;
pub use evidence::*;
pub use feature::*;
pub use ids::*;
pub use incremental_morphology::*;
pub use morphology::*;
pub use orthography::*;
pub use phonemicize::*;
pub use phonetics::*;
pub use phonology::*;
pub use plan_projection::*;
pub use prosody::*;
pub use realize::*;
pub use repair_delivery::*;
pub use rules::*;
pub use segment::*;
pub use spec::*;
pub use streaming::*;
pub use syllabify::*;
pub use syntax::*;
pub use text_stability::*;
pub use time::*;
pub use transcript::*;
pub use tts_ledger::*;
pub use utterance::*;
pub use variety::*;
#[cfg(feature = "asr-whisper")]
pub use whisper::*;
pub use word_stream::{
    AudioRef, BoundarySource, DEFAULT_WORD_STREAM_VARIETY, HeuristicAcousticWordBoundaryRefiner,
    NoopWordBoundaryRefiner, PronunciationLookupStatus, TimedWordStream, TranscriptWord,
    WordBoundaryRefiner, WordCommitment, WordId, WordNode,
    WordPronunciation as StreamWordPronunciation, WordStreamId, WordStreamSource, WordTextSpan,
    WordTiming, attach_default_pronunciations, attach_pronunciations_for_variety,
    transcript_to_energy_snapped_word_stream, transcript_to_energy_snapped_word_stream_for_variety,
    transcript_to_word_stream, transcript_to_word_stream_for_variety,
};

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
