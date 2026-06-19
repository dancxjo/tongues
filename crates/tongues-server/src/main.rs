use axum::{
    extract::State,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Command;
use tower_http::services::ServeDir;

const STYLE_VECTOR_DIMS: usize = 256;

#[derive(Clone)]
struct AppState {
    workspace_root: PathBuf,
}

#[tokio::main]
async fn main() {
    let workspace_root = std::env::current_dir().unwrap();
    let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("public");
    let state = AppState { workspace_root };

    let app = Router::new()
        .route("/api/emotions", get(get_emotions))
        .route("/api/speak", post(speak))
        .fallback_service(ServeDir::new(static_dir))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Web server listening on http://{}", addr);
    axum::serve(tokio::net::TcpListener::bind(&addr).await.unwrap(), app)
        .await
        .unwrap();
}

#[derive(Serialize)]
struct EmotionsResponse {
    signature_path: String,
    style_vectors_path: Option<String>,
    emotions: Vec<EmotionSignature>,
    generated_from_style_vectors: bool,
    error: Option<String>,
}

#[derive(Serialize, Clone)]
struct EmotionSignature {
    name: String,
    kind: String,
    method: String,
    dims: usize,
    vector: Vec<f32>,
    stats: EmotionStats,
    recommended_strength: RecommendedStrength,
}

#[derive(Serialize, Clone, Default)]
struct EmotionStats {
    n_speakers: usize,
    sample_count: usize,
}

#[derive(Serialize, Clone)]
struct RecommendedStrength {
    subtle: f32,
    normal: f32,
    strong: f32,
}

impl Default for RecommendedStrength {
    fn default() -> Self {
        Self {
            subtle: 0.25,
            normal: 0.65,
            strong: 1.10,
        }
    }
}

async fn get_emotions(State(state): State<AppState>) -> impl IntoResponse {
    match load_or_create_emotion_signatures(&state) {
        Ok(response) => Json(response).into_response(),
        Err(error) => {
            let signature_path = emotion_signatures_path(&state);
            Json(EmotionsResponse {
                signature_path: signature_path.display().to_string(),
                style_vectors_path: find_style_vectors_path(&state)
                    .map(|path| path.display().to_string()),
                emotions: Vec::new(),
                generated_from_style_vectors: false,
                error: Some(error),
            })
            .into_response()
        }
    }
}

#[derive(Deserialize)]
struct SpeakRequest {
    text: String,
    emotion: Option<String>,
    emotion_vector: Option<Vec<f32>>,
    emotion_strength: Option<f32>,
}

async fn speak(State(state): State<AppState>, Json(payload): Json<SpeakRequest>) -> impl IntoResponse {
    let out_wav = state.workspace_root.join(format!("output_{}.wav", uuid::Uuid::new_v4()));
    let temp_signatures = match write_request_emotion_signatures(&state, &payload) {
        Ok(path) => path,
        Err(error) => {
            return (axum::http::StatusCode::BAD_REQUEST, error).into_response();
        }
    };
    
    let mut args = vec![
        "run".to_string(),
        "--bin".to_string(),
        "tongues".to_string(),
        "--".to_string(),
        "speak".to_string(),
        "--output".to_string(),
        out_wav.to_string_lossy().to_string(),
    ];

    let emotion = payload.emotion.as_deref().unwrap_or_default();
    if !emotion.is_empty() {
        let signatures_path = temp_signatures
            .as_ref()
            .cloned()
            .unwrap_or_else(|| emotion_signatures_path(&state));
        if signatures_path.exists() {
            args.push("--emotion-signatures".to_string());
            args.push(signatures_path.to_string_lossy().to_string());
            args.push("--emotion".to_string());
            args.push(emotion.to_string());
            if let Some(strength) = payload.emotion_strength {
                args.push("--emotion-strength".to_string());
                args.push(strength.to_string());
            }
        }
    }

    // Pass the text as the final positional argument
    args.push(payload.text);

    let output = Command::new("cargo")
        .args(&args)
        .current_dir(&state.workspace_root)
        .output();

    match output {
        Ok(out) if out.status.success() => {
            if let Some(path) = temp_signatures {
                let _ = std::fs::remove_file(path);
            }
            if let Ok(audio_data) = std::fs::read(&out_wav) {
                let _ = std::fs::remove_file(&out_wav); // clean up
                Response::builder()
                    .header("Content-Type", "audio/wav")
                    .body(axum::body::Body::from(audio_data))
                    .unwrap()
            } else {
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to read generated audio").into_response()
            }
        }
        Ok(out) => {
            if let Some(path) = temp_signatures {
                let _ = std::fs::remove_file(path);
            }
            let stderr = String::from_utf8_lossy(&out.stderr);
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Synthesis failed: {}", stderr)).into_response()
        }
        Err(e) => {
            if let Some(path) = temp_signatures {
                let _ = std::fs::remove_file(path);
            }
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Command failed: {}", e)).into_response()
        }
    }
}

fn emotion_signatures_path(state: &AppState) -> PathBuf {
    state.workspace_root.join("emotion_signatures.json")
}

fn load_or_create_emotion_signatures(state: &AppState) -> Result<EmotionsResponse, String> {
    let signature_path = emotion_signatures_path(state);
    if signature_path.exists() {
        return load_emotion_signatures(state, false);
    }

    let Some(style_vectors_path) = find_style_vectors_path(state) else {
        return Ok(EmotionsResponse {
            signature_path: signature_path.display().to_string(),
            style_vectors_path: None,
            emotions: Vec::new(),
            generated_from_style_vectors: false,
            error: Some("No emotion_signatures.json or style_vectors.jsonl found".into()),
        });
    };

    let signatures = build_signatures_from_style_vectors(&style_vectors_path)?;
    write_emotion_signatures_file(&signature_path, &signatures)?;
    load_emotion_signatures(state, true)
}

fn load_emotion_signatures(
    state: &AppState,
    generated_from_style_vectors: bool,
) -> Result<EmotionsResponse, String> {
    let signature_path = emotion_signatures_path(state);
    let content = std::fs::read_to_string(&signature_path)
        .map_err(|error| format!("Failed to read {}: {error}", signature_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|error| format!("Failed to parse {}: {error}", signature_path.display()))?;
    let obj = json
        .as_object()
        .ok_or_else(|| "emotion_signatures.json must contain a JSON object".to_string())?;

    let sample_counts = find_style_vectors_path(state)
        .as_ref()
        .map(|path| load_emotion_sample_counts(path))
        .transpose()?
        .unwrap_or_default();

    let mut emotions = Vec::new();
    for (name, value) in obj {
        let vector = value
            .get("vector")
            .and_then(|vector| vector.as_array())
            .ok_or_else(|| format!("Emotion `{name}` is missing a vector array"))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|value| value as f32)
                    .ok_or_else(|| format!("Emotion `{name}` contains a non-numeric vector value"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_emotion_vector(name, &vector)?;

        let stats_value = value.get("stats");
        let n_speakers = stats_value
            .and_then(|stats| stats.get("n_speakers"))
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize;
        let recommended = value.get("recommended_strength");
        emotions.push(EmotionSignature {
            name: name.clone(),
            kind: value
                .get("kind")
                .and_then(|value| value.as_str())
                .unwrap_or("styletts2.emotion_signature.v1")
                .to_string(),
            method: value
                .get("method")
                .and_then(|value| value.as_str())
                .unwrap_or("speaker-neutral-delta")
                .to_string(),
            dims: value
                .get("dims")
                .and_then(|value| value.as_u64())
                .unwrap_or(vector.len() as u64) as usize,
            vector,
            stats: EmotionStats {
                n_speakers,
                sample_count: sample_counts.get(name).copied().unwrap_or(0),
            },
            recommended_strength: RecommendedStrength {
                subtle: recommended
                    .and_then(|value| value.get("subtle"))
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.25) as f32,
                normal: recommended
                    .and_then(|value| value.get("normal"))
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.65) as f32,
                strong: recommended
                    .and_then(|value| value.get("strong"))
                    .and_then(|value| value.as_f64())
                    .unwrap_or(1.10) as f32,
            },
        });
    }
    emotions.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(EmotionsResponse {
        signature_path: signature_path.display().to_string(),
        style_vectors_path: find_style_vectors_path(state).map(|path| path.display().to_string()),
        emotions,
        generated_from_style_vectors,
        error: None,
    })
}

fn find_style_vectors_path(state: &AppState) -> Option<PathBuf> {
    [
        state.workspace_root.join("style_vectors.jsonl"),
        state
            .workspace_root
            .join("datasets")
            .join("emotions")
            .join("style_vectors.jsonl"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

#[derive(Deserialize)]
struct StyleVectorEntry {
    emotion: String,
    speaker: String,
    vector: Vec<f32>,
}

fn build_signatures_from_style_vectors(
    path: &PathBuf,
) -> Result<BTreeMap<String, EmotionSignature>, String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("Failed to open {}: {error}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut speaker_map: HashMap<String, HashMap<String, Vec<Vec<f32>>>> = HashMap::new();

    for (line_index, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| {
            format!(
                "Failed to read line {} from {}: {error}",
                line_index + 1,
                path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: StyleVectorEntry = serde_json::from_str(&line).map_err(|error| {
            format!(
                "Failed to parse line {} from {}: {error}",
                line_index + 1,
                path.display()
            )
        })?;
        validate_emotion_vector(&entry.emotion, &entry.vector)?;
        speaker_map
            .entry(entry.speaker)
            .or_default()
            .entry(entry.emotion)
            .or_default()
            .push(entry.vector);
    }

    let mut emotion_deltas: BTreeMap<String, Vec<Vec<f32>>> = BTreeMap::new();
    for emotions in speaker_map.values() {
        let Some(neutrals) = emotions.get("neutral") else {
            continue;
        };
        let neutral_mean = mean_vector(neutrals);
        for (emotion, vectors) in emotions {
            if emotion == "neutral" {
                continue;
            }
            let emotion_mean = mean_vector(vectors);
            let delta = emotion_mean
                .iter()
                .zip(&neutral_mean)
                .map(|(emotion, neutral)| emotion - neutral)
                .collect::<Vec<_>>();
            emotion_deltas.entry(emotion.clone()).or_default().push(delta);
        }
    }

    let sample_counts = load_emotion_sample_counts(path)?;
    let mut signatures = BTreeMap::new();
    for (emotion, deltas) in emotion_deltas {
        let speakers = deltas.len();
        let vector = mean_vector(&deltas);
        signatures.insert(
            emotion.clone(),
            EmotionSignature {
                name: emotion.clone(),
                kind: "styletts2.emotion_signature.v1".into(),
                method: "speaker-neutral-delta".into(),
                dims: STYLE_VECTOR_DIMS,
                vector,
                stats: EmotionStats {
                    n_speakers: speakers,
                    sample_count: sample_counts.get(&emotion).copied().unwrap_or(0),
                },
                recommended_strength: RecommendedStrength::default(),
            },
        );
    }

    Ok(signatures)
}

fn write_emotion_signatures_file(
    path: &PathBuf,
    signatures: &BTreeMap<String, EmotionSignature>,
) -> Result<(), String> {
    let mut map = serde_json::Map::new();
    for (emotion, signature) in signatures {
        map.insert(
            emotion.clone(),
            json!({
                "kind": signature.kind,
                "emotion": signature.name,
                "method": signature.method,
                "dims": signature.dims,
                "vector": signature.vector,
                "stats": {
                    "n_speakers": signature.stats.n_speakers,
                    "sample_count": signature.stats.sample_count,
                },
                "recommended_strength": {
                    "subtle": signature.recommended_strength.subtle,
                    "normal": signature.recommended_strength.normal,
                    "strong": signature.recommended_strength.strong,
                },
            }),
        );
    }

    let part_path = path.with_extension("json.part");
    let file = std::fs::File::create(&part_path)
        .map_err(|error| format!("Failed to create {}: {error}", part_path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &serde_json::Value::Object(map))
        .map_err(|error| format!("Failed to write {}: {error}", part_path.display()))?;
    writeln!(writer).map_err(|error| format!("Failed to write {}: {error}", part_path.display()))?;
    writer
        .flush()
        .map_err(|error| format!("Failed to flush {}: {error}", part_path.display()))?;
    std::fs::rename(&part_path, path).map_err(|error| {
        format!(
            "Failed to rename {} to {}: {error}",
            part_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn load_emotion_sample_counts(path: &PathBuf) -> Result<HashMap<String, usize>, String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("Failed to open {}: {error}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut counts = HashMap::new();
    for line in reader.lines() {
        let line = line.map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .map_err(|error| format!("Failed to parse {}: {error}", path.display()))?;
        if let Some(emotion) = value.get("emotion").and_then(|value| value.as_str()) {
            *counts.entry(emotion.to_string()).or_insert(0) += 1;
        }
    }
    Ok(counts)
}

fn mean_vector(vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut mean = vec![0.0; STYLE_VECTOR_DIMS];
    for vector in vectors {
        for (index, value) in vector.iter().enumerate().take(STYLE_VECTOR_DIMS) {
            mean[index] += value;
        }
    }
    for value in &mut mean {
        *value /= vectors.len() as f32;
    }
    mean
}

fn validate_emotion_vector(name: &str, vector: &[f32]) -> Result<(), String> {
    if vector.len() != STYLE_VECTOR_DIMS {
        return Err(format!(
            "Emotion `{name}` vector must have {STYLE_VECTOR_DIMS} values, got {}",
            vector.len()
        ));
    }
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(format!("Emotion `{name}` vector contains non-finite values"));
    }
    Ok(())
}

fn write_request_emotion_signatures(
    state: &AppState,
    payload: &SpeakRequest,
) -> Result<Option<PathBuf>, String> {
    let Some(vector) = payload.emotion_vector.as_ref() else {
        return Ok(None);
    };
    let emotion = payload
        .emotion
        .as_deref()
        .filter(|emotion| !emotion.is_empty())
        .ok_or_else(|| "emotion is required when emotion_vector is provided".to_string())?;
    validate_emotion_vector(emotion, vector)?;

    let path = state
        .workspace_root
        .join(format!("emotion_request_{}.json", uuid::Uuid::new_v4()));
    let signature = json!({
        emotion: {
            "kind": "styletts2.emotion_signature.v1",
            "emotion": emotion,
            "method": "frontend-posted-delta",
            "dims": STYLE_VECTOR_DIMS,
            "vector": vector,
            "stats": {
                "n_speakers": 0,
                "sample_count": 0,
            },
            "recommended_strength": RecommendedStrength::default(),
        }
    });
    let file = std::fs::File::create(&path)
        .map_err(|error| format!("Failed to create {}: {error}", path.display()))?;
    serde_json::to_writer(file, &signature)
        .map_err(|error| format!("Failed to write {}: {error}", path.display()))?;
    Ok(Some(path))
}
