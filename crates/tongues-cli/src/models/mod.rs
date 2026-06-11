pub mod cli;
mod download;
mod manifest;
mod selection;

pub use cli::{run, ModelsCommand};
pub use download::{
    ensure_asr_whisper_model_available, ensure_face_models_available, ensure_model_available,
    ensure_piper_voice_model_available, ensure_runtime_models_available,
    ensure_selected_llm_available, ensure_styletts2_default_reference_audio_available,
    ensure_styletts2_model_available, missing_model_asset_paths, FaceModelPaths, RuntimeModelPaths,
    StyleTts2ReferenceAudioPaths,
};
pub use manifest::{
    bundle_multimodal_projector_asset, ModelAsset, ModelBundle, DEFAULT_ASR_MODEL_ID,
    DEFAULT_FACE_MODEL_ID, DEFAULT_LLM_MODEL_ID, DEFAULT_PIPER_VOICE_MODEL_ID,
    DEFAULT_STYLETTS2_MODEL_ID, MODEL_ASSETS, MODEL_BUNDLES,
};
pub use selection::{
    selected_bundle, selected_bundle_for_kind, selected_llm_model_label, selected_llm_model_path,
    selected_piper_voice_bundle,
};
