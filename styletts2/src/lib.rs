//! StyleTTS2-native synthesis seam for Pete Mortar-Sea.
//!
//! This crate owns the StyleTTS2-facing model/config/input-output contract
//! without making StyleTTS2 the owner of speech inside Pete Mortar-Sea. The intended
//! path is not text-to-speech first:
//!
//! ```text
//! UtterancePlan
//!   -> variety-aware phones/phonemes
//!   -> speaker identity
//!   -> style reference
//!   -> StyleTTS2 backend
//!   -> waveform
//!   -> observed/realized Utterance
//! ```
//!
//! `speaker` and `style` are deliberately separate. Speaker identity answers
//! who is speaking; style references answer how they are speaking. Phones and
//! phonemes carry what is being said, and the prosody track carries timing,
//! energy, and pitch intent.
//!
//! Real inference belongs behind [`StyleTts2Backend`]. ONNX accelerator support
//! is enabled by default, while tests use the deterministic mock and contract
//! paths unless they explicitly load a native backend.

pub mod backend;
pub mod config;
pub mod mock;
pub mod plan;
pub mod request;
pub mod symbols;

pub use backend::{
    StyleTts2AudioChunk, StyleTts2AudioSink, StyleTts2Backend, StyleTts2Error,
    StyleTts2SynthesisOutput, StyleTts2Timing,
};
pub use config::{StyleTts2Config, StyleTts2ConfigError, StyleTts2ModelPaths};
pub use mock::MockStyleTts2Backend;
pub use plan::{
    BackendSynthesisPlan, DEFAULT_MAX_TTS_SYMBOLS, StyleTts2PlanOptions, SynthesisChunk,
    prepare_styletts2_plan, styletts2_text_for_symbols, styletts2_text_to_ids,
    validate_styletts2_plan,
};
pub use request::StyleTts2SynthesisRequest;
pub use symbols::{
    StyleTts2SymbolMapper, StyleTts2SymbolSequence, StyleTts2SymbolSource, StyleTts2SymbolToken,
    SymbolLoweringError, SymbolSet, styletts2_en_us_symbol_set,
};

#[cfg(feature = "styletts2-onnx")]
pub mod onnx;

#[cfg(feature = "styletts2-onnx")]
pub use onnx::{
    STYLETTS2_ONNX_INTER_THREADS_ENV, STYLETTS2_ONNX_INTRA_THREADS_ENV, StyleTts2DiffusionOptions,
    StyleTts2OnnxBackend, StyleTts2OnnxOptimization, StyleTts2OnnxOptions, StyleTts2OnnxPaths,
};
