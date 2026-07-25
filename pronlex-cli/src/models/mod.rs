pub mod cli;
pub mod download;
pub mod manifest;
pub mod selection;

pub use cli::{ModelsCommand, run};
pub use download::{
    ensure_voice_model_available, ensure_styletts2_model_available,
    ensure_styletts2_default_reference_audio_available, styletts2_default_reference_audio_paths,
    missing_model_asset_paths,
};
pub use manifest::{
    DEFAULT_VOICE_MODEL_ID, DEFAULT_STYLETTS2_MODEL_ID, MODEL_ASSETS, MODEL_BUNDLES,
    ModelAsset, ModelBundle,
};
pub use selection::{
    selected_voice_model_bundle, selected_bundle_for_kind, resolve_pronlex_home,
};
