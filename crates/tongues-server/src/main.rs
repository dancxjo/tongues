use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::net::SocketAddr;
use std::path::{Component, Path as FsPath, PathBuf};
use std::process::Command;
use tower_http::services::ServeDir;

const STYLE_VECTOR_DIMS: usize = 256;
const STYLETTS2_REFERENCE_RELATIVE_DIR: &str = "models/styletts2/en-us/reference_audio";

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
        .route("/api/styletts2-samples", get(get_styletts2_samples))
        .route(
            "/api/styletts2-reference-audio/{*sample_id}",
            get(get_styletts2_reference_audio),
        )
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

#[derive(Serialize)]
struct StyleTts2SamplesResponse {
    reference_dir: Option<String>,
    samples: Vec<StyleTts2Sample>,
    defaults: StyleTts2SampleDefaults,
    error: Option<String>,
}

#[derive(Serialize, Clone)]
struct StyleTts2Sample {
    id: String,
    label: String,
    path: String,
    audio_url: String,
    duration_ms: Option<u64>,
}

#[derive(Serialize)]
struct StyleTts2SampleDefaults {
    voice: String,
    style: String,
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
    voice_sample: Option<String>,
    style_sample: Option<String>,
    quality: Option<String>,
    diffusion_steps: Option<usize>,
    speaker_reference_strength: Option<f32>,
    style_reference_strength: Option<f32>,
    style_alpha: Option<f32>,
    style_beta: Option<f32>,
    embedding_scale: Option<f64>,
    style_seed: Option<u64>,
    speed: Option<f64>,
    sample_rate_hz: Option<u32>,
    max_tts_symbols: Option<usize>,
    no_tts_chunking: Option<bool>,
}

async fn speak(State(state): State<AppState>, Json(payload): Json<SpeakRequest>) -> impl IntoResponse {
    if let Err(error) = validate_speak_request(&payload) {
        return (StatusCode::BAD_REQUEST, error).into_response();
    }

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

    if let Some(sample_rate_hz) = payload.sample_rate_hz {
        args.push("--sample-rate-hz".to_string());
        args.push(sample_rate_hz.to_string());
    }

    if let Some(quality) = payload.quality.as_deref().filter(|value| !value.is_empty()) {
        args.push("--quality".to_string());
        args.push(quality.to_string());
    }

    if let Some(diffusion_steps) = payload.diffusion_steps {
        args.push("--diffusion-steps".to_string());
        args.push(diffusion_steps.to_string());
    }

    if let Some(voice_sample) = payload.voice_sample.as_deref().filter(|value| !value.is_empty()) {
        match styletts2_sample_path(&state, voice_sample) {
            Ok(path) => {
                args.push("--voice-wav".to_string());
                args.push(path.to_string_lossy().to_string());
            }
            Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
        }
    }

    if let Some(style_sample) = payload.style_sample.as_deref().filter(|value| !value.is_empty()) {
        match styletts2_sample_path(&state, style_sample) {
            Ok(path) => {
                args.push("--style-wav".to_string());
                args.push(path.to_string_lossy().to_string());
            }
            Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
        }
    }

    if let Some(strength) = payload.speaker_reference_strength {
        args.push("--speaker-reference-strength".to_string());
        args.push(strength.to_string());
    }

    if let Some(strength) = payload.style_reference_strength {
        args.push("--style-reference-strength".to_string());
        args.push(strength.to_string());
    }

    if let Some(alpha) = payload.style_alpha {
        args.push("--style-alpha".to_string());
        args.push(alpha.to_string());
    }

    if let Some(beta) = payload.style_beta {
        args.push("--style-beta".to_string());
        args.push(beta.to_string());
    }

    if let Some(embedding_scale) = payload.embedding_scale {
        args.push("--embedding-scale".to_string());
        args.push(embedding_scale.to_string());
    }

    if let Some(style_seed) = payload.style_seed {
        args.push("--style-seed".to_string());
        args.push(style_seed.to_string());
    }

    if let Some(speed) = payload.speed {
        args.push("--speed".to_string());
        args.push(speed.to_string());
    }

    if let Some(max_tts_symbols) = payload.max_tts_symbols {
        args.push("--max-tts-symbols".to_string());
        args.push(max_tts_symbols.to_string());
    }

    if payload.no_tts_chunking.unwrap_or(false) {
        args.push("--no-tts-chunking".to_string());
    }

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

async fn get_styletts2_samples(State(state): State<AppState>) -> impl IntoResponse {
    let response = match load_styletts2_samples(&state) {
        Ok(samples) => StyleTts2SamplesResponse {
            reference_dir: Some(styletts2_reference_dir(&state).display().to_string()),
            samples,
            defaults: StyleTts2SampleDefaults {
                voice: "1221-135767-0014.wav".into(),
                style: "amused.wav".into(),
            },
            error: None,
        },
        Err(error) => StyleTts2SamplesResponse {
            reference_dir: Some(styletts2_reference_dir(&state).display().to_string()),
            samples: Vec::new(),
            defaults: StyleTts2SampleDefaults {
                voice: "1221-135767-0014.wav".into(),
                style: "amused.wav".into(),
            },
            error: Some(error),
        },
    };
    Json(response)
}

async fn get_styletts2_reference_audio(
    State(state): State<AppState>,
    Path(sample_id): Path<String>,
) -> impl IntoResponse {
    let path = match styletts2_sample_path(&state, &sample_id) {
        Ok(path) => path,
        Err(error) => return (StatusCode::BAD_REQUEST, error).into_response(),
    };
    match std::fs::read(&path) {
        Ok(bytes) => Response::builder()
            .header("Content-Type", "audio/wav")
            .body(axum::body::Body::from(bytes))
            .unwrap(),
        Err(error) => (
            StatusCode::NOT_FOUND,
            format!("Failed to read {}: {error}", path.display()),
        )
            .into_response(),
    }
}

fn emotion_signatures_path(state: &AppState) -> PathBuf {
    state.workspace_root.join("emotion_signatures.json")
}

fn validate_speak_request(payload: &SpeakRequest) -> Result<(), String> {
    if payload.text.trim().is_empty() {
        return Err("text is required".into());
    }
    if let Some(quality) = payload.quality.as_deref() {
        if !quality.is_empty() && quality != "balanced" && quality != "fast" {
            return Err("quality must be `balanced` or `fast`".into());
        }
    }
    if let Some(diffusion_steps) = payload.diffusion_steps {
        if !(1..=64).contains(&diffusion_steps) {
            return Err("diffusion_steps must be between 1 and 64".into());
        }
    }
    validate_f32_range(
        "speaker_reference_strength",
        payload.speaker_reference_strength,
        0.0,
        1.0,
    )?;
    validate_f32_range(
        "style_reference_strength",
        payload.style_reference_strength,
        0.0,
        1.0,
    )?;
    validate_f32_range("style_alpha", payload.style_alpha, 0.0, 1.0)?;
    validate_f32_range("style_beta", payload.style_beta, 0.0, 1.0)?;
    validate_f64_range("embedding_scale", payload.embedding_scale, 0.0, 5.0)?;
    validate_f64_range("speed", payload.speed, 0.25, 3.0)?;
    if let Some(sample_rate_hz) = payload.sample_rate_hz {
        if !(8_000..=48_000).contains(&sample_rate_hz) {
            return Err("sample_rate_hz must be between 8000 and 48000".into());
        }
    }
    if let Some(max_tts_symbols) = payload.max_tts_symbols {
        if !(16..=2048).contains(&max_tts_symbols) {
            return Err("max_tts_symbols must be between 16 and 2048".into());
        }
    }
    Ok(())
}

fn validate_f32_range(name: &str, value: Option<f32>, min: f32, max: f32) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() || value < min || value > max {
        return Err(format!("{name} must be a finite value from {min} to {max}"));
    }
    Ok(())
}

fn validate_f64_range(name: &str, value: Option<f64>, min: f64, max: f64) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() || value < min || value > max {
        return Err(format!("{name} must be a finite value from {min} to {max}"));
    }
    Ok(())
}

fn styletts2_reference_dir(_state: &AppState) -> PathBuf {
    resolve_mortar_home()
        .join(STYLETTS2_REFERENCE_RELATIVE_DIR)
        .canonicalize()
        .unwrap_or_else(|_| resolve_mortar_home().join(STYLETTS2_REFERENCE_RELATIVE_DIR))
}

fn resolve_mortar_home() -> PathBuf {
    if let Some(home) = std::env::var_os("MORTAR_SEA_HOME") {
        let home = PathBuf::from(home);
        if !home.as_os_str().is_empty() {
            return home;
        }
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("mortar-sea")
}

fn load_styletts2_samples(state: &AppState) -> Result<Vec<StyleTts2Sample>, String> {
    let reference_dir = styletts2_reference_dir(state);
    if !reference_dir.is_dir() {
        return Err(format!(
            "StyleTTS2 reference audio is not extracted at {}. Run `cargo run --bin tongues -- models fetch styletts2` or synthesize once to download it.",
            reference_dir.display()
        ));
    }

    let mut samples = Vec::new();
    collect_wav_samples(&reference_dir, &reference_dir, &mut samples)?;
    samples.sort_by(|a, b| a.label.cmp(&b.label).then_with(|| a.id.cmp(&b.id)));
    Ok(samples)
}

fn collect_wav_samples(
    reference_dir: &FsPath,
    dir: &FsPath,
    samples: &mut Vec<StyleTts2Sample>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| format!("Failed to read {}: {error}", dir.display()))?;
    for entry in entries {
        let entry =
            entry.map_err(|error| format!("Failed to read entry in {}: {error}", dir.display()))?;
        let path = entry.path();
        let metadata = entry
            .metadata()
            .map_err(|error| format!("Failed to read metadata for {}: {error}", path.display()))?;
        if metadata.is_dir() {
            collect_wav_samples(reference_dir, &path, samples)?;
            continue;
        }
        if !metadata.is_file() || !is_wav_path(&path) {
            continue;
        }
        let relative = path
            .strip_prefix(reference_dir)
            .map_err(|error| format!("Failed to relativize {}: {error}", path.display()))?;
        let id = relative_path_id(relative)?;
        samples.push(StyleTts2Sample {
            label: sample_label(relative),
            audio_url: format!("/api/styletts2-reference-audio/{}", url_path_escape(&id)),
            path: path.display().to_string(),
            duration_ms: wav_duration_ms(&path),
            id,
        });
    }
    Ok(())
}

fn is_wav_path(path: &FsPath) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
}

fn relative_path_id(path: &FsPath) -> Result<String, String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let part = part
                    .to_str()
                    .ok_or_else(|| "sample path contains non-UTF-8 data".to_string())?;
                parts.push(part.to_string());
            }
            _ => return Err("sample path contains invalid components".into()),
        }
    }
    Ok(parts.join("/"))
}

fn sample_label(path: &FsPath) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("sample")
        .replace(['_', '-'], " ")
}

fn url_path_escape(path: &str) -> String {
    path.split('/')
        .map(|part| {
            part.bytes()
                .flat_map(|byte| match byte {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'-' | b'_' | b'~' => {
                        vec![byte as char]
                    }
                    _ => format!("%{byte:02X}").chars().collect(),
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn styletts2_sample_path(state: &AppState, sample_id: &str) -> Result<PathBuf, String> {
    if sample_id.trim().is_empty() {
        return Err("sample id is required".into());
    }
    let relative = FsPath::new(sample_id);
    for component in relative.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err("sample id must be a relative path under reference_audio".into());
        }
    }
    if !is_wav_path(relative) {
        return Err("sample id must point to a WAV file".into());
    }

    let reference_dir = styletts2_reference_dir(state);
    let path = reference_dir.join(relative);
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("Unknown StyleTTS2 sample `{sample_id}`: {error}"))?;
    let canonical_reference_dir = reference_dir
        .canonicalize()
        .map_err(|error| format!("StyleTTS2 reference directory is unavailable: {error}"))?;
    if !canonical.starts_with(&canonical_reference_dir) || !canonical.is_file() {
        return Err("sample id is outside the StyleTTS2 reference directory".into());
    }
    Ok(canonical)
}

fn wav_duration_ms(path: &FsPath) -> Option<u64> {
    let reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return None;
    }
    let samples_per_channel = reader.duration() / u32::from(spec.channels);
    Some((u64::from(samples_per_channel) * 1_000) / u64::from(spec.sample_rate))
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
