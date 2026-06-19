use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use clap::ValueEnum;
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EmotionCorpusArg {
    Ravdess,
    CremaD,
    Tess,
    Savee,
    Emodb,
    Iemocap,
}

impl EmotionCorpusArg {
    fn all() -> &'static [Self] {
        &[
            Self::Ravdess,
            Self::CremaD,
            Self::Tess,
            Self::Savee,
            Self::Emodb,
            Self::Iemocap,
        ]
    }

    fn id(self) -> &'static str {
        match self {
            Self::Ravdess => "ravdess",
            Self::CremaD => "crema-d",
            Self::Tess => "tess",
            Self::Savee => "savee",
            Self::Emodb => "emodb",
            Self::Iemocap => "iemocap",
        }
    }

    fn dir_name(self) -> &'static str {
        match self {
            Self::Ravdess => "RAVDESS",
            Self::CremaD => "CREMA-D",
            Self::Tess => "TESS",
            Self::Savee => "SAVEE",
            Self::Emodb => "EmoDB",
            Self::Iemocap => "IEMOCAP",
        }
    }
}

#[derive(Debug, Clone)]
struct LabelRow {
    path: PathBuf,
    emotion: String,
    speaker: String,
    corpus: &'static str,
    extras: serde_json::Map<String, serde_json::Value>,
}

pub fn cmd_fetch_corpora(out_dir: &Path, corpora: &[EmotionCorpusArg], list: bool) -> Result<()> {
    if list {
        print_available_corpora();
        return Ok(());
    }

    let selected = if corpora.is_empty() {
        EmotionCorpusArg::all().to_vec()
    } else {
        corpora.to_vec()
    };

    println!("Fetching emotion corpora to {}...", out_dir.display());
    println!(
        "Selected corpora: {}",
        selected
            .iter()
            .map(|corpus| corpus.id())
            .collect::<Vec<_>>()
            .join(", ")
    );

    fs::create_dir_all(out_dir).context("failed to create output directory")?;

    for corpus in &selected {
        match corpus {
            EmotionCorpusArg::Ravdess => fetch_ravdess(out_dir)?,
            EmotionCorpusArg::CremaD => fetch_crema_d(out_dir)?,
            EmotionCorpusArg::Tess => print_manual_tess(out_dir),
            EmotionCorpusArg::Savee => print_manual_savee(out_dir),
            EmotionCorpusArg::Emodb => fetch_emodb(out_dir)?,
            EmotionCorpusArg::Iemocap => print_manual_iemocap(out_dir),
        }
    }

    write_labels(out_dir, &selected)?;
    Ok(())
}

fn print_available_corpora() {
    println!("Available emotion corpora:");
    for corpus in EmotionCorpusArg::all() {
        println!("  {:<8} {}", corpus.id(), corpus_summary(*corpus));
    }
    println!();
    println!("Use --corpus more than once to pick a subset, for example:");
    println!("  tongues fetch-corpora --corpus ravdess --corpus crema-d --corpus emodb");
}

fn corpus_summary(corpus: EmotionCorpusArg) -> &'static str {
    match corpus {
        EmotionCorpusArg::Ravdess => "direct Zenodo ZIP; acted English speech",
        EmotionCorpusArg::CremaD => "Git LFS clone; acted English speech",
        EmotionCorpusArg::Tess => "manual download; clean acted English speech",
        EmotionCorpusArg::Savee => "manual download; British English acted speech",
        EmotionCorpusArg::Emodb => "direct Zenodo ZIP; acted German speech",
        EmotionCorpusArg::Iemocap => "licensed/manual download; conversational emotion",
    }
}

fn fetch_ravdess(out_dir: &Path) -> Result<()> {
    let archive = out_dir.join("RAVDESS_Audio_Speech_Actors_01-24.zip");
    let dir = out_dir.join(EmotionCorpusArg::Ravdess.dir_name());
    fetch_zip_if_missing(
        "RAVDESS",
        "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
        &archive,
        &dir,
    )
}

fn fetch_emodb(out_dir: &Path) -> Result<()> {
    let archive = out_dir.join("emodb_2.0.zip");
    let dir = out_dir.join(EmotionCorpusArg::Emodb.dir_name());
    fetch_zip_if_missing(
        "EmoDB",
        "https://zenodo.org/records/17651657/files/emodb_2.0.zip?download=1",
        &archive,
        &dir,
    )
}

fn fetch_crema_d(out_dir: &Path) -> Result<()> {
    let dir = out_dir.join(EmotionCorpusArg::CremaD.dir_name());
    if dir.exists() {
        println!("CREMA-D already present at {}", dir.display());
        return Ok(());
    }

    println!("Fetching CREMA-D with git-lfs into {}...", dir.display());
    let status = Command::new("git")
        .args([
            "lfs",
            "clone",
            "https://github.com/CheyneyComputerScience/CREMA-D.git",
            dir.to_str().context("CREMA-D path is not valid UTF-8")?,
        ])
        .status();

    match status {
        Ok(status) if status.success() => Ok(()),
        _ => {
            println!(
                "Warning: CREMA-D needs git-lfs. Install git-lfs, then run:\n  git lfs clone https://github.com/CheyneyComputerScience/CREMA-D.git {}",
                dir.display()
            );
            Ok(())
        }
    }
}

fn fetch_zip_if_missing(name: &str, url: &str, archive: &Path, dir: &Path) -> Result<()> {
    if !archive.exists() {
        download(url, archive).unwrap_or_else(|err| {
            println!("Warning: failed to download {name}: {err}");
        });
    } else {
        println!("{name} archive already present at {}", archive.display());
    }

    if archive.exists() && !dir.exists() {
        println!("Extracting {name} to {}...", dir.display());
        fs::create_dir_all(dir)?;
        let status = Command::new("unzip")
            .args([
                "-q",
                archive
                    .to_str()
                    .context("archive path is not valid UTF-8")?,
                "-d",
                dir.to_str().context("extract path is not valid UTF-8")?,
            ])
            .status();
        match status {
            Ok(status) if status.success() => {}
            Ok(status) => println!("Warning: unzip for {name} exited with {status}"),
            Err(err) => println!("Warning: failed to run unzip for {name}: {err}"),
        }
    } else if dir.exists() {
        println!("{name} already extracted at {}", dir.display());
    }

    Ok(())
}

fn download(url: &str, dest: &Path) -> Result<()> {
    let part_path = dest.with_extension(format!(
        "{}part",
        dest.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| format!("{ext}."))
            .unwrap_or_default()
    ));
    let _ = fs::remove_file(&part_path);

    println!("Downloading {url}");
    let dest_arg = part_path
        .to_str()
        .context("download path is not valid UTF-8")?;
    let curl = Command::new("curl")
        .args(["-fL", "--retry", "3", "-o", dest_arg, url])
        .status();
    let ok = match curl {
        Ok(status) if status.success() => true,
        _ => {
            let wget = Command::new("wget")
                .args(["-O", dest_arg, url])
                .status()
                .context("neither curl nor wget could be started")?;
            wget.success()
        }
    };
    anyhow::ensure!(ok, "failed to download {url}");
    fs::rename(&part_path, dest)
        .with_context(|| format!("failed to move {} into place", dest.display()))?;
    Ok(())
}

fn print_manual_tess(out_dir: &Path) {
    let target = out_dir.join(EmotionCorpusArg::Tess.dir_name());
    println!(
        "TESS requires manual download. Put the extracted Toronto emotional speech set under {}.",
        target.display()
    );
    println!("Known source: https://utoronto.scholaris.ca/collections/036db644-9790-4ed0-90cc-be1dfb8a4b66");
}

fn print_manual_savee(out_dir: &Path) {
    let target = out_dir.join(EmotionCorpusArg::Savee.dir_name());
    println!(
        "SAVEE requires manual download. Put the extracted SAVEE audio under {}.",
        target.display()
    );
    println!("Known source: https://kahlan.eps.surrey.ac.uk/savee/");
}

fn print_manual_iemocap(out_dir: &Path) {
    let target = out_dir.join(EmotionCorpusArg::Iemocap.dir_name());
    println!(
        "IEMOCAP is licensed/manual access. Put the extracted release under {}.",
        target.display()
    );
    println!("Known source: https://sail.usc.edu/iemocap/iemocap_release.htm");
}

fn write_labels(out_dir: &Path, selected: &[EmotionCorpusArg]) -> Result<()> {
    let mut labels = Vec::new();
    for corpus in selected {
        let dir = out_dir.join(corpus.dir_name());
        if !dir.exists() {
            continue;
        }
        let before = labels.len();
        match corpus {
            EmotionCorpusArg::Ravdess => collect_ravdess(&dir, &mut labels)?,
            EmotionCorpusArg::CremaD => collect_crema_d(&dir, &mut labels)?,
            EmotionCorpusArg::Tess => collect_tess(&dir, &mut labels)?,
            EmotionCorpusArg::Savee => collect_savee(&dir, &mut labels)?,
            EmotionCorpusArg::Emodb => collect_emodb(&dir, &mut labels)?,
            EmotionCorpusArg::Iemocap => collect_iemocap(&dir, &mut labels)?,
        }
        println!(
            "Labeled {:>5} rows from {}",
            labels.len() - before,
            corpus.id()
        );
    }

    let labels_path = out_dir.join("labels.jsonl");
    let part_path = out_dir.join("labels.jsonl.part");
    println!("Writing {}...", labels_path.display());
    let mut out = BufWriter::new(File::create(&part_path)?);
    for label in labels {
        let mut row = label.extras;
        row.insert(
            "path".to_string(),
            json!(label
                .path
                .canonicalize()
                .unwrap_or(label.path)
                .display()
                .to_string()),
        );
        row.insert("emotion".to_string(), json!(label.emotion));
        row.insert("speaker".to_string(), json!(label.speaker));
        row.insert("corpus".to_string(), json!(label.corpus));
        writeln!(out, "{}", serde_json::Value::Object(row))?;
    }
    out.flush()?;
    fs::rename(&part_path, &labels_path)?;
    println!(
        "Done. Use `tongues styletts2 encode-style` with {}.",
        labels_path.display()
    );
    Ok(())
}

fn collect_ravdess(root: &Path, labels: &mut Vec<LabelRow>) -> Result<()> {
    let emotions = [
        "",
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    ];
    for wav in wavs(root) {
        let Some(stem) = wav.file_stem().and_then(|name| name.to_str()) else {
            continue;
        };
        let parts = stem.split('-').collect::<Vec<_>>();
        if parts.len() != 7 {
            continue;
        }
        let emotion_idx = parts[2].parse::<usize>().unwrap_or(0);
        if emotion_idx == 0 || emotion_idx >= emotions.len() {
            continue;
        }
        let actor = parts[6].parse::<usize>().unwrap_or(0);
        let intensity = match parts[3] {
            "01" => "normal",
            "02" => "strong",
            _ => "unknown",
        };
        labels.push(label(
            wav,
            emotions[emotion_idx],
            format!("ravdess_{actor:02}"),
            "ravdess",
            [("intensity", intensity.to_string())],
        ));
    }
    Ok(())
}

fn collect_crema_d(root: &Path, labels: &mut Vec<LabelRow>) -> Result<()> {
    for wav in wavs(root) {
        let Some(stem) = wav.file_stem().and_then(|name| name.to_str()) else {
            continue;
        };
        let parts = stem.split('_').map(str::to_string).collect::<Vec<_>>();
        if parts.len() < 4 {
            continue;
        }
        let Some(emotion) = crema_emotion(&parts[2]) else {
            continue;
        };
        labels.push(label(
            wav,
            emotion,
            format!("cremad_{}", parts[0]),
            "crema-d",
            [("intensity", crema_intensity(&parts[3]).to_string())],
        ));
    }
    Ok(())
}

fn collect_tess(root: &Path, labels: &mut Vec<LabelRow>) -> Result<()> {
    for wav in wavs(root) {
        let Some(stem) = wav.file_stem().and_then(|name| name.to_str()) else {
            continue;
        };
        let lower = stem.to_ascii_lowercase();
        let emotion = if lower.contains("pleasant_surprise") || lower.ends_with("_ps") {
            Some("surprised")
        } else {
            lower.rsplit('_').next().and_then(|raw| simple_emotion(raw))
        };
        let Some(emotion) = emotion else {
            continue;
        };
        let speaker = stem
            .split('_')
            .next()
            .filter(|prefix| !prefix.is_empty())
            .unwrap_or("unknown")
            .to_string();
        labels.push(label(
            wav,
            emotion,
            format!("tess_{}", speaker.to_ascii_lowercase()),
            "tess",
            [],
        ));
    }
    Ok(())
}

fn collect_savee(root: &Path, labels: &mut Vec<LabelRow>) -> Result<()> {
    for wav in wavs(root) {
        let Some(stem) = wav.file_stem().and_then(|name| name.to_str()) else {
            continue;
        };
        let parts = stem.split('_').map(str::to_string).collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        let raw = parts[1].trim_end_matches(|c: char| c.is_ascii_digit());
        let Some(emotion) = savee_emotion(raw) else {
            continue;
        };
        labels.push(label(
            wav,
            emotion,
            format!("savee_{}", parts[0].to_ascii_lowercase()),
            "savee",
            [],
        ));
    }
    Ok(())
}

fn collect_emodb(root: &Path, labels: &mut Vec<LabelRow>) -> Result<()> {
    for wav in wavs(root) {
        let Some(stem) = wav.file_stem().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(code) = stem.chars().last() else {
            continue;
        };
        let Some(emotion) = emodb_emotion(code) else {
            continue;
        };
        let speaker = stem.get(0..2).unwrap_or("unknown").to_string();
        labels.push(label(wav, emotion, format!("emodb_{speaker}"), "emodb", []));
    }
    Ok(())
}

fn collect_iemocap(root: &Path, labels: &mut Vec<LabelRow>) -> Result<()> {
    let mut utterance_emotions = HashMap::new();
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("txt") {
            continue;
        }
        if !path.to_string_lossy().contains("EmoEvaluation") {
            continue;
        }
        let file = File::open(path)?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            if !line.starts_with('[') {
                continue;
            }
            let cols = line.split('\t').collect::<Vec<_>>();
            if cols.len() < 3 {
                continue;
            }
            if let Some(emotion) = iemocap_emotion(cols[2].trim()) {
                utterance_emotions.insert(cols[1].trim().to_string(), emotion.to_string());
            }
        }
    }

    for wav in wavs(root) {
        let Some(stem) = wav.file_stem().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(emotion) = utterance_emotions.get(stem) else {
            continue;
        };
        let speaker = stem
            .rsplit('_')
            .next()
            .and_then(|suffix| suffix.chars().next())
            .map(|speaker| format!("iemocap_{speaker}"))
            .unwrap_or_else(|| "iemocap_unknown".to_string());
        labels.push(label(wav, emotion, speaker, "iemocap", []));
    }
    Ok(())
}

fn wavs(root: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.into_path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
        })
        .collect()
}

fn label<const N: usize>(
    path: PathBuf,
    emotion: &str,
    speaker: String,
    corpus: &'static str,
    extras: [(&str, String); N],
) -> LabelRow {
    LabelRow {
        path,
        emotion: emotion.to_string(),
        speaker,
        corpus,
        extras: extras
            .into_iter()
            .map(|(key, value)| (key.to_string(), json!(value)))
            .collect(),
    }
}

fn crema_emotion(raw: &str) -> Option<&'static str> {
    match raw {
        "ANG" => Some("angry"),
        "DIS" => Some("disgust"),
        "FEA" => Some("fearful"),
        "HAP" => Some("happy"),
        "NEU" => Some("neutral"),
        "SAD" => Some("sad"),
        _ => None,
    }
}

fn crema_intensity(raw: &str) -> &'static str {
    match raw {
        "LO" => "low",
        "MD" => "medium",
        "HI" => "high",
        "XX" => "unspecified",
        _ => "unknown",
    }
}

fn simple_emotion(raw: &str) -> Option<&'static str> {
    match raw {
        "angry" | "anger" => Some("angry"),
        "disgust" | "disgusted" => Some("disgust"),
        "fear" | "fearful" => Some("fearful"),
        "happy" | "happiness" | "joy" => Some("happy"),
        "neutral" => Some("neutral"),
        "sad" | "sadness" => Some("sad"),
        "surprise" | "surprised" => Some("surprised"),
        "calm" => Some("calm"),
        "frustrated" | "frustration" => Some("frustrated"),
        _ => None,
    }
}

fn savee_emotion(raw: &str) -> Option<&'static str> {
    match raw {
        "a" => Some("angry"),
        "d" => Some("disgust"),
        "f" => Some("fearful"),
        "h" => Some("happy"),
        "n" => Some("neutral"),
        "sa" => Some("sad"),
        "su" => Some("surprised"),
        _ => None,
    }
}

fn emodb_emotion(raw: char) -> Option<&'static str> {
    match raw {
        'W' | 'w' => Some("angry"),
        'L' | 'l' => Some("boredom"),
        'E' | 'e' => Some("disgust"),
        'A' | 'a' => Some("fearful"),
        'F' | 'f' => Some("happy"),
        'T' | 't' => Some("sad"),
        'N' | 'n' => Some("neutral"),
        _ => None,
    }
}

fn iemocap_emotion(raw: &str) -> Option<&'static str> {
    match raw {
        "ang" => Some("angry"),
        "hap" | "exc" => Some("happy"),
        "sad" => Some("sad"),
        "fru" => Some("frustrated"),
        "neu" => Some("neutral"),
        "fea" => Some("fearful"),
        "sur" => Some("surprised"),
        "dis" => Some("disgust"),
        _ => None,
    }
}
