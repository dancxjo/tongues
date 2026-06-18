use std::fs;
use std::path::Path;
use std::process::Command;
use anyhow::{Context, Result};

pub fn cmd_fetch_corpora(out_dir: &Path) -> Result<()> {
    println!("Fetching emotion corpora to {}...", out_dir.display());

    if !out_dir.exists() {
        fs::create_dir_all(out_dir).context("failed to create output directory")?;
    }

    // A small helper to run curl/wget
    let fetch = |url: &str, dest: &Path| -> Result<()> {
        println!("Downloading {}", url);
        let status = Command::new("curl")
            .args(["-fsSL", "-o", dest.to_str().unwrap(), url])
            .status();
        match status {
            Ok(s) if s.success() => Ok(()),
            _ => {
                let s = Command::new("wget")
                    .args(["-qO", dest.to_str().unwrap(), url])
                    .status()
                    .context("neither curl nor wget succeeded")?;
                if s.success() {
                    Ok(())
                } else {
                    anyhow::bail!("Failed to download {}", url)
                }
            }
        }
    };

    // RAVDESS Audio (Speech)
    let ravdess_zip = out_dir.join("RAVDESS_Audio_Speech_Actors_01-24.zip");
    if !ravdess_zip.exists() {
        let url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip";
        fetch(url, &ravdess_zip).unwrap_or_else(|e| println!("Warning: {}", e));
    }

    let ravdess_dir = out_dir.join("RAVDESS");
    if ravdess_zip.exists() && !ravdess_dir.exists() {
        println!("Extracting RAVDESS...");
        fs::create_dir_all(&ravdess_dir)?;
        let s = Command::new("unzip")
            .args(["-q", ravdess_zip.to_str().unwrap(), "-d", ravdess_dir.to_str().unwrap()])
            .status();
        if let Err(e) = s {
            println!("Warning: failed to unzip RAVDESS: {}", e);
        }
    }

    // Generate labels.jsonl for RAVDESS
    let labels_path = out_dir.join("labels.jsonl");
    println!("Generating {}...", labels_path.display());
    use std::io::Write;
    let mut labels_out = std::io::BufWriter::new(fs::File::create(&labels_path)?);

    // Parse RAVDESS filenames
    // Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    // Vocal channel (01 = speech, 02 = song).
    // Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    // Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    // Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    // Repetition (01 = 1st repetition, 02 = 2nd repetition).
    // Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    if ravdess_dir.exists() {
        let emotions = ["", "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"];
        for entry in walkdir::WalkDir::new(&ravdess_dir).into_iter().filter_map(|e| e.ok()) {
            if entry.path().extension().and_then(|e| e.to_str()) == Some("wav") {
                let filename = entry.file_name().to_string_lossy();
                let parts: Vec<&str> = filename.trim_end_matches(".wav").split('-').collect();
                if parts.len() == 7 {
                    let emotion_idx: usize = parts[2].parse().unwrap_or(0);
                    let actor_idx: usize = parts[6].parse().unwrap_or(0);
                    if emotion_idx > 0 && emotion_idx < emotions.len() {
                        let emotion = emotions[emotion_idx];
                        let json = serde_json::json!({
                            "path": entry.path().canonicalize().unwrap_or(entry.path().to_path_buf()).display().to_string(),
                            "emotion": emotion,
                            "speaker": format!("ravdess_{:02}", actor_idx)
                        });
                        writeln!(labels_out, "{}", serde_json::to_string(&json)?)?;
                    }
                }
            }
        }
    }

    labels_out.flush()?;
    println!("Done. You can now use `tongues styletts2 encode-style` with {}", labels_path.display());
    Ok(())
}
