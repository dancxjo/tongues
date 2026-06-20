use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const CMUDICT_URL: &str = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict";
const CMUDICT_SOURCE: &str = "src/data/lexicons/cmudict.dict";
const LEXIQUE383_URL: &str = "http://www.lexique.org/databases/Lexique383/Lexique383.tsv";
const LEXIQUE383_SOURCE: &str = "src/data/lexicons/Lexique383.tsv";
const LEXIQUE383_SEED: &str = "\
ortho\tphon
avant\tavɑ̃
bagne\tbaɲ
bonjour\tbɔ̃ʒuʁ
cela\tsəla
et\te
français\tfʁɑ̃sɛ
intelligent\tɛ̃teliʒɑ̃
j'étais\tʒetɛ
paysan\tpeizɑ̃
pauvre\tpovʁ
peu\tpø
recueillez\tʁəkœje
si\tsi
très\ttrɛ
voulez\tvule
vous\tvu
";

fn main() {
    let whisper_enabled = std::env::var_os("CARGO_FEATURE_ASR_WHISPER").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").ok();
    if whisper_enabled && target_os.as_deref() == Some("linux") {
        println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
        println!("cargo:rustc-link-lib=gomp");
    }

    println!("cargo:rerun-if-changed={CMUDICT_SOURCE}");
    println!("cargo:rerun-if-changed={LEXIQUE383_SOURCE}");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let bundled_cmudict = out_dir.join("cmudict.dict");
    ensure_cmudict(&bundled_cmudict);
    let bundled_lexique = out_dir.join("Lexique383.tsv");
    ensure_lexique383(&bundled_lexique);
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

fn ensure_lexique383(out_path: &Path) {
    if out_path.exists() {
        return;
    }

    if Path::new(LEXIQUE383_SOURCE).exists() {
        fs::copy(LEXIQUE383_SOURCE, out_path).expect("copying bundled Lexique383.tsv into OUT_DIR");
        return;
    }

    let tmp_path = out_path.with_extension("part");
    let url = LEXIQUE383_URL;
    let download_status = Command::new("curl")
        .args([
            "-fsSL",
            "--max-time",
            "120",
            "-o",
            tmp_path.to_str().expect("valid temp path"),
            url,
        ])
        .status()
        .or_else(|_| {
            Command::new("wget")
                .args([
                    "-T",
                    "120",
                    "-qO",
                    tmp_path.to_str().expect("valid temp path"),
                    url,
                ])
                .status()
        });

    if matches!(download_status, Ok(status) if status.success()) {
        fs::rename(&tmp_path, out_path).expect("publishing downloaded Lexique383.tsv into OUT_DIR");
    } else {
        let _ = fs::remove_file(&tmp_path);
        fs::write(out_path, LEXIQUE383_SEED).expect("writing seed Lexique383.tsv into OUT_DIR");
    }
}
