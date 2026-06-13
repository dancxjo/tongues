fn main() {
    let whisper_enabled = std::env::var_os("CARGO_FEATURE_ASR_WHISPER").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").ok();
    if whisper_enabled && target_os.as_deref() == Some("linux") {
        println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
        println!("cargo:rustc-link-lib=gomp");
    }
}
