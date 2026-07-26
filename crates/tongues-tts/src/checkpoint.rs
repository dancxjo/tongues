use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn_store::{
    ApplyResult, ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
};

pub(crate) struct CheckpointLoadOptions<'a> {
    pub top_level_key: Option<&'a str>,
    pub predicate: Option<fn(&str, &str) -> bool>,
    pub key_remappings: Vec<(String, String)>,
    pub map_indices_contiguous: bool,
    pub allow_partial: bool,
    pub skip_enum_variants: bool,
}

impl Default for CheckpointLoadOptions<'_> {
    fn default() -> Self {
        Self {
            top_level_key: None,
            predicate: None,
            key_remappings: Vec::new(),
            map_indices_contiguous: true,
            allow_partial: false,
            skip_enum_variants: false,
        }
    }
}

pub(crate) fn load_pytorch_layout_checkpoint<B: Backend, M: ModuleSnapshot<B>>(
    module: &mut M,
    path: &Path,
    options: CheckpointLoadOptions<'_>,
) -> Result<ApplyResult> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
    {
        let mut store = SafetensorsStore::from_file(path)
            .with_from_adapter(PyTorchToBurnAdapter)
            .skip_enum_variants(options.skip_enum_variants)
            .map_indices_contiguous(options.map_indices_contiguous)
            .allow_partial(options.allow_partial);
        for (pattern, replacement) in &options.key_remappings {
            store = store.with_key_remapping(pattern, replacement);
        }
        if let Some(predicate) = options.predicate {
            store = store.with_predicate(predicate);
        }
        module
            .load_from(&mut store)
            .with_context(|| format!("failed to load SafeTensors checkpoint {}", path.display()))
    } else {
        let mut store = PytorchStore::from_file(path)
            .skip_enum_variants(options.skip_enum_variants)
            .map_indices_contiguous(options.map_indices_contiguous)
            .allow_partial(options.allow_partial);
        if let Some(key) = options.top_level_key {
            store = store.with_top_level_key(key);
        }
        for (pattern, replacement) in &options.key_remappings {
            store = store.with_key_remapping(pattern, replacement);
        }
        if let Some(predicate) = options.predicate {
            store = store.with_predicate(predicate);
        }
        module
            .load_from(&mut store)
            .with_context(|| format!("failed to load PyTorch checkpoint {}", path.display()))
    }
}
