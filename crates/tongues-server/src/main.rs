use axum::{
    extract::{Query, State},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use tower_http::services::ServeDir;

#[derive(Clone)]
struct AppState {
    workspace_root: PathBuf,
}

#[tokio::main]
async fn main() {
    let workspace_root = std::env::current_dir().unwrap();
    let state = AppState { workspace_root };

    let app = Router::new()
        .route("/api/emotions", get(get_emotions))
        .route("/api/speak", post(speak))
        .nest_service("/", ServeDir::new("public"))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Web server listening on http://{}", addr);
    axum::serve(tokio::net::TcpListener::bind(&addr).await.unwrap(), app)
        .await
        .unwrap();
}

#[derive(Serialize)]
struct EmotionsResponse {
    emotions: Vec<String>,
}

async fn get_emotions(State(state): State<AppState>) -> impl IntoResponse {
    let sig_path = state.workspace_root.join("emotion_signatures.json");
    if let Ok(content) = std::fs::read_to_string(&sig_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(obj) = json.as_object() {
                let emotions: Vec<String> = obj.keys().cloned().collect();
                return Json(EmotionsResponse { emotions }).into_response();
            }
        }
    }
    Json(EmotionsResponse { emotions: vec![] }).into_response()
}

#[derive(Deserialize)]
struct SpeakRequest {
    text: String,
    emotion: Option<String>,
    emotion_strength: Option<f32>,
}

async fn speak(State(state): State<AppState>, Json(payload): Json<SpeakRequest>) -> impl IntoResponse {
    let out_wav = state.workspace_root.join(format!("output_{}.wav", uuid::Uuid::new_v4()));
    
    let mut args = vec![
        "run".to_string(),
        "--bin".to_string(),
        "tongues".to_string(),
        "--".to_string(),
        "speak".to_string(),
        "--output".to_string(),
        out_wav.to_string_lossy().to_string(),
    ];

    if let Some(em) = payload.emotion {
        if !em.is_empty() {
            args.push("--emotion-signatures".to_string());
            args.push("emotion_signatures.json".to_string());
            args.push("--emotion".to_string());
            args.push(em);
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
            let stderr = String::from_utf8_lossy(&out.stderr);
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Synthesis failed: {}", stderr)).into_response()
        }
        Err(e) => {
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Command failed: {}", e)).into_response()
        }
    }
}
