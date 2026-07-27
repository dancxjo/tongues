use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use speaking::{SegmentId, StreamEvent, TextRole};
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use tokio::sync::mpsc;

const DEFAULT_OLLAMA_HOST: &str = "http://127.0.0.1:11434";
const MIN_CLAUSE_CHARS: usize = 48;
const MAX_SEGMENT_CHARS: usize = 64;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeechInstruction {
    pub language: Option<String>,
    pub variety: Option<String>,
    pub script: Option<String>,
    pub normalization: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatTurnRequest {
    pub turn_id: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub response_instructions: String,
    pub speech: Option<SpeechInstruction>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveProvider {
    pub id: &'static str,
    pub label: &'static str,
    pub available: bool,
    pub models: Vec<String>,
    pub detail: String,
}

pub trait StreamingTextProvider: Send + Sync {
    fn stream_turn<'a>(
        &'a self,
        request: &'a ChatTurnRequest,
        events: mpsc::Sender<StreamEvent>,
        cancelled: Arc<AtomicBool>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send + 'a>>;
}

pub struct OllamaProvider {
    client: reqwest::Client,
    host: String,
}

impl OllamaProvider {
    pub fn from_environment() -> anyhow::Result<Self> {
        let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_OLLAMA_HOST.into());
        let host = host.trim_end_matches('/').to_string();
        if !(host.starts_with("http://") || host.starts_with("https://")) {
            anyhow::bail!("OLLAMA_HOST must use http:// or https://");
        }
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(10 * 60))
            .build()?;
        Ok(Self { client, host })
    }

    pub async fn models(&self) -> anyhow::Result<Vec<String>> {
        #[derive(Deserialize)]
        struct Tags {
            #[serde(default)]
            models: Vec<Tag>,
        }
        #[derive(Deserialize)]
        struct Tag {
            model: String,
            #[serde(default)]
            details: TagDetails,
        }
        #[derive(Default, Deserialize)]
        struct TagDetails {
            #[serde(default)]
            family: String,
        }
        let response = self
            .client
            .get(format!("{}/api/tags", self.host))
            .timeout(Duration::from_secs(2))
            .send()
            .await?
            .error_for_status()?;
        let mut models = response
            .json::<Tags>()
            .await?
            .models
            .into_iter()
            .filter(|model| {
                !model.model.to_ascii_lowercase().contains("embed")
                    && !model.details.family.to_ascii_lowercase().contains("embed")
            })
            .map(|model| model.model)
            .collect::<Vec<_>>();
        models.sort();
        Ok(models)
    }
}

impl StreamingTextProvider for OllamaProvider {
    // Keep the nested stream validation readable on the repository's Rust 2021
    // edition; equivalent let-chain syntax requires a newer edition.
    #[allow(clippy::collapsible_if)]
    fn stream_turn<'a>(
        &'a self,
        request: &'a ChatTurnRequest,
        events: mpsc::Sender<StreamEvent>,
        cancelled: Arc<AtomicBool>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            #[derive(Deserialize)]
            struct OllamaChunk {
                #[serde(default)]
                message: Option<ChatMessage>,
                #[serde(default)]
                done: bool,
                error: Option<String>,
            }

            let mut messages = request.messages.clone();
            if let Some(instruction) = generator_instruction(request) {
                messages.insert(
                    0,
                    ChatMessage {
                        role: "system".into(),
                        content: instruction,
                    },
                );
            }
            let mut response = self
                .client
                .post(format!("{}/api/chat", self.host))
                .json(&serde_json::json!({
                    "model": request.model,
                    "messages": messages,
                    "stream": true,
                }))
                .send()
                .await
                .context("connecting to Ollama")?
                .error_for_status()
                .context("Ollama rejected the chat request")?;
            let mut buffered = Vec::new();
            let mut generated = String::new();
            while let Some(chunk) = response.chunk().await? {
                if cancelled.load(Ordering::Acquire) {
                    return Ok(());
                }
                buffered.extend_from_slice(&chunk);
                while let Some(newline) = buffered.iter().position(|byte| *byte == b'\n') {
                    let line = buffered.drain(..=newline).collect::<Vec<_>>();
                    let line = std::str::from_utf8(&line)?.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let chunk: OllamaChunk =
                        serde_json::from_str(line).context("decoding Ollama stream chunk")?;
                    if let Some(error) = chunk.error {
                        anyhow::bail!("Ollama stream failed: {error}");
                    }
                    if let Some(message) = chunk.message {
                        if !message.content.is_empty() {
                            generated.push_str(&message.content);
                            if events
                                .send(StreamEvent::PartialHypothesis {
                                    role: TextRole::Generation,
                                    segment_id: SegmentId(format!(
                                        "generation-{}",
                                        request.turn_id
                                    )),
                                    text: generated.clone(),
                                    confidence: None,
                                })
                                .await
                                .is_err()
                            {
                                return Ok(());
                            }
                        }
                    }
                    if chunk.done {
                        return Ok(());
                    }
                }
            }
            Ok(())
        })
    }
}

pub struct DeterministicProvider;

impl StreamingTextProvider for DeterministicProvider {
    fn stream_turn<'a>(
        &'a self,
        request: &'a ChatTurnRequest,
        events: mpsc::Sender<StreamEvent>,
        cancelled: Arc<AtomicBool>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let prompt = request
                .messages
                .iter()
                .rev()
                .find(|message| message.role == "user")
                .map(|message| message.content.as_str())
                .unwrap_or("speech");
            let topic = prompt.chars().take(24).collect::<String>();
            let response = format!(
                "Speech starts now. I am answering about {topic}. \
                 More words are still arriving. Generation and playback continue together."
            );
            let mut generated = String::new();
            for token in response.split_inclusive(' ') {
                if cancelled.load(Ordering::Acquire) {
                    break;
                }
                generated.push_str(token);
                if events
                    .send(StreamEvent::PartialHypothesis {
                        role: TextRole::Generation,
                        segment_id: SegmentId(format!("generation-{}", request.turn_id)),
                        text: generated.clone(),
                        confidence: None,
                    })
                    .await
                    .is_err()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(35)).await;
            }
            Ok(())
        })
    }
}

pub async fn provider_discovery() -> Vec<LiveProvider> {
    let ollama = match OllamaProvider::from_environment() {
        Ok(provider) => match provider.models().await {
            Ok(models) => LiveProvider {
                id: "ollama",
                label: "Ollama",
                available: true,
                detail: if models.is_empty() {
                    "Ollama is reachable; pull a model to begin.".into()
                } else {
                    "Local Ollama token streaming.".into()
                },
                models,
            },
            Err(error) => LiveProvider {
                id: "ollama",
                label: "Ollama",
                available: false,
                models: Vec::new(),
                detail: format!("Ollama is unavailable: {error:#}"),
            },
        },
        Err(error) => LiveProvider {
            id: "ollama",
            label: "Ollama",
            available: false,
            models: Vec::new(),
            detail: error.to_string(),
        },
    };
    vec![
        ollama,
        LiveProvider {
            id: "deterministic",
            label: "Deterministic test stream",
            available: true,
            models: vec!["fixture-live-v1".into()],
            detail: "Repeatable local stream for testing the complete queue.".into(),
        },
    ]
}

pub fn spawn_turn(
    request: ChatTurnRequest,
    cancelled: Arc<AtomicBool>,
) -> mpsc::Receiver<StreamEvent> {
    let (turn_tx, turn_rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let _ = turn_tx
            .send(StreamEvent::SessionStarted {
                purpose: "generated_speech_turn".into(),
            })
            .await;
        let (provider_tx, mut provider_rx) = mpsc::channel(32);
        let provider: Box<dyn StreamingTextProvider> = match request.provider.as_str() {
            "ollama" => match OllamaProvider::from_environment() {
                Ok(provider) => Box::new(provider),
                Err(error) => {
                    let _ = send_failure(&turn_tx, &request.turn_id, error.to_string()).await;
                    return;
                }
            },
            "deterministic" => Box::new(DeterministicProvider),
            other => {
                let _ = send_failure(
                    &turn_tx,
                    &request.turn_id,
                    format!("unknown provider `{other}`"),
                )
                .await;
                return;
            }
        };
        let provider_request = request.clone();
        let provider_cancelled = Arc::clone(&cancelled);
        let provider_task = tokio::spawn(async move {
            provider
                .stream_turn(&provider_request, provider_tx, provider_cancelled)
                .await
        });
        let mut generated = String::new();
        let mut committed = String::new();
        let mut segmenter = StreamingSegmenter::default();
        let mut segment_id = 0;
        while let Some(event) = provider_rx.recv().await {
            if cancelled.load(Ordering::Acquire) {
                break;
            }
            let StreamEvent::PartialHypothesis {
                role: TextRole::Generation,
                ref text,
                ..
            } = event
            else {
                if turn_tx.send(event).await.is_err() {
                    cancelled.store(true, Ordering::Release);
                    return;
                }
                continue;
            };
            let Some(delta) = text.strip_prefix(&generated).map(str::to_owned) else {
                let _ = send_failure(
                    &turn_tx,
                    &request.turn_id,
                    "text provider revised already streamed generation".into(),
                )
                .await;
                return;
            };
            generated = text.clone();
            if turn_tx.send(event).await.is_err() {
                cancelled.store(true, Ordering::Release);
                return;
            }
            for segment in segmenter.push(&delta) {
                segment_id += 1;
                committed.push_str(&segment);
                let event = StreamEvent::CommittedSegment {
                    role: TextRole::Generation,
                    segment_id: SegmentId(format!("generation-segment-{segment_id}")),
                    text: segment,
                    words: Vec::new(),
                    language: None,
                    speaker_id: None,
                    confidence: None,
                };
                if turn_tx.send(event).await.is_err() {
                    cancelled.store(true, Ordering::Release);
                    return;
                }
            }
        }
        let provider_result = provider_task.await;
        if cancelled.load(Ordering::Acquire) {
            for segment in segmenter.finish() {
                committed.push_str(&segment);
            }
            let _ = turn_tx
                .send(StreamEvent::Cancelled {
                    reason: format!(
                        "turn {} cancelled after {} generated and {} committed characters",
                        request.turn_id,
                        generated.chars().count(),
                        committed.chars().count()
                    ),
                })
                .await;
            return;
        }
        match provider_result {
            Ok(Ok(())) => {
                for segment in segmenter.finish() {
                    segment_id += 1;
                    committed.push_str(&segment);
                    if turn_tx
                        .send(StreamEvent::CommittedSegment {
                            role: TextRole::Generation,
                            segment_id: SegmentId(format!("generation-segment-{segment_id}")),
                            text: segment,
                            words: Vec::new(),
                            language: None,
                            speaker_id: None,
                            confidence: None,
                        })
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                let _ = turn_tx
                    .send(StreamEvent::TextCompleted {
                        role: TextRole::Generation,
                        text: generated,
                    })
                    .await;
            }
            Ok(Err(error)) => {
                let _ = send_failure(&turn_tx, &request.turn_id, format!("{error:#}")).await;
            }
            Err(error) => {
                let _ = send_failure(
                    &turn_tx,
                    &request.turn_id,
                    format!("provider task failed: {error}"),
                )
                .await;
            }
        }
    });
    turn_rx
}

async fn send_failure(
    sink: &mpsc::Sender<StreamEvent>,
    _turn_id: &str,
    message: String,
) -> Result<(), mpsc::error::SendError<StreamEvent>> {
    sink.send(StreamEvent::Error {
        code: "provider_failure".into(),
        message,
        recoverable: false,
    })
    .await
}

pub fn generator_instruction(request: &ChatTurnRequest) -> Option<String> {
    let mut requirements = Vec::new();
    if let Some(speech) = &request.speech {
        if let Some(language) = nonempty(&speech.language) {
            requirements.push(format!("Respond in {language}"));
        }
        if let Some(variety) = nonempty(&speech.variety) {
            requirements.push(format!("use the {variety} language or variety"));
        }
        if let Some(script) = nonempty(&speech.script) {
            requirements.push(format!("write in {script} script"));
        }
        if let Some(normalization) = nonempty(&speech.normalization) {
            requirements.push(format!(
                "follow this speech normalization requirement: {normalization}"
            ));
        }
    }
    if !request.response_instructions.trim().is_empty() {
        requirements.push(request.response_instructions.trim().into());
    }
    (!requirements.is_empty()).then(|| {
        format!(
            "You are writing text that will be spoken immediately. {}. \
             Do not mention these instructions. Prefer complete, naturally speakable sentences.",
            requirements.join("; ")
        )
    })
}

fn nonempty(value: &Option<String>) -> Option<&str> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

#[derive(Default)]
pub struct StreamingSegmenter {
    pending: String,
}

impl StreamingSegmenter {
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        self.pending.push_str(delta);
        let mut committed = Vec::new();
        while let Some(boundary) = find_boundary(&self.pending) {
            let remainder = self.pending.split_off(boundary);
            let segment = std::mem::replace(&mut self.pending, remainder);
            if !segment.trim().is_empty() {
                committed.push(segment);
            }
        }
        committed
    }

    pub fn finish(&mut self) -> Vec<String> {
        let final_segment = std::mem::take(&mut self.pending);
        if final_segment.trim().is_empty() {
            Vec::new()
        } else {
            vec![final_segment]
        }
    }
}

fn find_boundary(text: &str) -> Option<usize> {
    let chars = text.char_indices().collect::<Vec<_>>();
    let abbreviations = [
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "e.g.", "i.e.",
    ];
    let mut quote_depth = false;
    let mut nesting = 0_i32;
    let mut clause = None;
    let mut soft = None;
    for (position, &(byte, character)) in chars.iter().enumerate() {
        match character {
            '"' | '“' | '”' | '«' | '»' => quote_depth = !quote_depth,
            '(' | '[' | '{' => nesting += 1,
            ')' | ']' | '}' => nesting = (nesting - 1).max(0),
            _ => {}
        }
        let char_count = position + 1;
        let end = byte + character.len_utf8();
        if character.is_whitespace() && char_count >= MAX_SEGMENT_CHARS && soft.is_none() {
            soft = Some(end);
        }
        if !quote_depth
            && nesting == 0
            && matches!(character, ',' | ';' | ':' | '，' | '；' | '：')
            && char_count >= MIN_CLAUSE_CHARS
        {
            clause = Some(end);
        }
        if matches!(character, '.' | '!' | '?' | '。' | '！' | '？' | '።') {
            let previous = position.checked_sub(1).and_then(|index| chars.get(index));
            let next = chars.get(position + 1);
            let decimal = previous.is_some_and(|(_, value)| value.is_ascii_digit())
                && next.is_some_and(|(_, value)| value.is_ascii_digit());
            let closes_quote = next.is_some_and(|(_, value)| {
                matches!(value, '"' | '”' | '»' | '\'' | ')' | ']' | '}')
            });
            let prefix = text[..end].trim_end().to_ascii_lowercase();
            let abbreviation = abbreviations.iter().any(|item| prefix.ends_with(item));
            if !decimal
                && !abbreviation
                && (!quote_depth || closes_quote)
                && (nesting == 0 || closes_quote)
            {
                if char_count > MAX_SEGMENT_CHARS
                    && let Some(soft_boundary) = soft
                {
                    return Some(soft_boundary);
                }
                let mut boundary = end;
                for &(next_byte, next_char) in chars.iter().skip(position + 1) {
                    if matches!(next_char, '"' | '”' | '»' | '\'' | ')' | ']' | '}')
                        || next_char.is_whitespace()
                    {
                        boundary = next_byte + next_char.len_utf8();
                    } else {
                        break;
                    }
                }
                return Some(boundary);
            }
        }
    }
    clause.or(soft)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn segmenter_commits_terminal_sentences_without_reordering() {
        let mut segmenter = StreamingSegmenter::default();
        let mut segments = Vec::new();
        for delta in ["One sentence. ", "Two ", "sentences! And three"] {
            segments.extend(segmenter.push(delta));
        }
        segments.extend(segmenter.finish());
        assert_eq!(segments, ["One sentence. ", "Two sentences! ", "And three"]);
        assert_eq!(segments.concat(), "One sentence. Two sentences! And three");
    }

    #[test]
    fn segmenter_avoids_decimal_abbreviation_and_unfinished_quote_splits() {
        let mut segmenter = StreamingSegmenter::default();
        assert!(
            segmenter
                .push("Dr. Ada said, \"Use 3.14, not 3.")
                .is_empty()
        );
        assert_eq!(
            segmenter.push("0.\" Then she left. "),
            ["Dr. Ada said, \"Use 3.14, not 3.0.\" ", "Then she left. "]
        );
        assert!(segmenter.finish().is_empty());
    }

    #[test]
    fn speech_recipe_constrains_generator_language_and_script() {
        let request = ChatTurnRequest {
            turn_id: "turn-1".into(),
            provider: "deterministic".into(),
            model: "fixture-live-v1".into(),
            messages: Vec::new(),
            response_instructions: "Tell a short story.".into(),
            speech: Some(SpeechInstruction {
                language: Some("Amharic".into()),
                variety: Some("am".into()),
                script: Some("Ethiopic".into()),
                normalization: None,
            }),
        };
        let instruction = generator_instruction(&request).unwrap();
        assert!(instruction.contains("Respond in Amharic"));
        assert!(instruction.contains("Ethiopic script"));
        assert!(instruction.contains("Tell a short story"));
    }

    #[test]
    fn every_generated_character_is_committed_exactly_once() {
        let input = "First phrase, with enough material to commit safely after the clause; second phrase ends here.";
        let mut segmenter = StreamingSegmenter::default();
        let mut output = Vec::new();
        for delta in input.as_bytes().chunks(7) {
            output.extend(segmenter.push(std::str::from_utf8(delta).unwrap()));
        }
        output.extend(segmenter.finish());
        assert_eq!(output.concat(), input);
        assert_eq!(output.iter().collect::<BTreeSet<_>>().len(), output.len());
    }

    #[tokio::test]
    async fn deterministic_turn_streams_text_before_generation_finishes() {
        let request = ChatTurnRequest {
            turn_id: "turn-stream-order".into(),
            provider: "deterministic".into(),
            model: "fixture-live-v1".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "streaming speech".into(),
            }],
            response_instructions: String::new(),
            speech: None,
        };
        let mut events = spawn_turn(request, Arc::new(AtomicBool::new(false)));
        let mut generated = String::new();
        let mut committed = String::new();
        let mut saw_delta = false;
        let mut saw_segment_before_done = false;
        while let Some(event) = events.recv().await {
            match event {
                StreamEvent::PartialHypothesis {
                    role: TextRole::Generation,
                    text,
                    ..
                } => {
                    saw_delta = true;
                    generated = text;
                }
                StreamEvent::CommittedSegment {
                    role: TextRole::Generation,
                    text,
                    ..
                } => {
                    assert!(saw_delta);
                    saw_segment_before_done = true;
                    committed.push_str(&text);
                }
                StreamEvent::TextCompleted {
                    role: TextRole::Generation,
                    text,
                } => {
                    assert!(saw_segment_before_done);
                    assert_eq!(text, generated);
                    assert_eq!(committed, generated);
                    return;
                }
                _ => {}
            }
        }
        panic!("deterministic turn ended without text_completed");
    }

    #[test]
    fn audio_event_serializes_as_a_typed_stream_record() {
        let event = StreamEvent::AudioChunk {
            direction: speaking::AudioDirection::Output,
            chunk_sequence: 0,
            frame_count: 12_000,
            segment_id: Some(SegmentId("generation-segment-2".into())),
            format: Some(speaking::AudioFormat {
                encoding: speaking::AudioEncoding::Wav,
                sample_rate_hz: 24_000,
                channels: speaking::ChannelLayout::Mono,
            }),
            audio_base64: Some("UklGRg==".into()),
            metadata: std::collections::BTreeMap::from([(
                "resident_model_reused".into(),
                serde_json::Value::Bool(true),
            )]),
        };
        let value = serde_json::to_value(event).unwrap();
        assert_eq!(value["type"], "audio_chunk");
        assert_eq!(value["data"]["direction"], "output");
        assert_eq!(value["data"]["format"]["encoding"], "wav");
    }
}
