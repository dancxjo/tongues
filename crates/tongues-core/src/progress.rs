use std::sync::OnceLock;

static MULTI_PROGRESS: OnceLock<indicatif::MultiProgress> = OnceLock::new();

/// Get the global MultiProgress instance.
pub fn get_multi_progress() -> &'static indicatif::MultiProgress {
    MULTI_PROGRESS.get_or_init(indicatif::MultiProgress::new)
}

/// Register a ProgressBar with the global MultiProgress instance.
pub fn register_progress_bar(pb: indicatif::ProgressBar) -> indicatif::ProgressBar {
    if !pb.is_hidden() {
        get_multi_progress().add(pb)
    } else {
        pb
    }
}
