use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const CMUDICT_URL: &str = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict";
const CMUDICT_SOURCE: &str = "src/data/lexicons/cmudict.dict";

fn main() {
    let whisper_enabled = std::env::var_os("CARGO_FEATURE_ASR_WHISPER").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").ok();
    if whisper_enabled && target_os.as_deref() == Some("linux") {
        println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
        println!("cargo:rustc-link-lib=gomp");
    }

    println!("cargo:rerun-if-changed={CMUDICT_SOURCE}");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let bundled_cmudict = out_dir.join("cmudict.dict");
    ensure_cmudict(&bundled_cmudict);
}

fn ensure_cmudict(out_path: &Path) {
    if out_path.exists() {
        return;
    }

    if let Some(source_path) = Path::new(CMUDICT_SOURCE)
        .parent()
        .map(|_| Path::new(CMUDICT_SOURCE))
        && source_path.exists()
    {
        fs::copy(source_path, out_path).expect("copying bundled cmudict.dict into OUT_DIR");
        return;
    }

    let tmp_path = out_path.with_extension("part");
    let url = CMUDICT_URL;

    let download_status = Command::new("curl")
        .args([
            "-fsSL",
            "-o",
            tmp_path.to_str().expect("valid temp path"),
            url,
        ])
        .status()
        .or_else(|_| {
            Command::new("wget")
                .args(["-qO", tmp_path.to_str().expect("valid temp path"), url])
                .status()
        })
        .expect("starting a downloader for cmudict.dict");

    if !download_status.success() {
        panic!("failed to download cmudict.dict from {url}");
    }

    fs::rename(&tmp_path, out_path).expect("publishing downloaded cmudict.dict into OUT_DIR");
}
