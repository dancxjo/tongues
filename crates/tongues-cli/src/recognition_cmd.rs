use std::collections::BTreeSet;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{ArgGroup, Args, ValueEnum};
use speaking::{
    committed_transcript, recognition_workflow, AsrDecodingControl, AsrProviderCapabilities,
    AsrResourceLimits, AsrRuntime, AsrSessionConfig, AsrStreamingCapability,
    CommittedTranscriptPipeline, FixtureAsrProvider, FixtureAsrStep, FriendlySpeechVerb,
    LanguageId, RuleBasedTranscriptNormalizer, StreamEvent, StructuralTranscriptInterpreter,
    TranscriptSourceMetadata, WhisperAsrProvider,
};
use tongues_audio::{
    AudioSource, AudioSourceDescriptor, AudioSourceEvent, CpalAudioSource, NormalizedAudioSource,
    PcmEncoding, PcmReaderSource, WavAudioSource,
};

const ASR_SAMPLE_RATE_HZ: u32 = 16_000;
const SOURCE_CHUNK_FRAMES: usize = 1_600;
const RAW_READ_BYTES: usize = 6_400;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RecognitionOutput {
    Text,
    Json,
    Jsonl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RecognitionProvider {
    Whisper,
    Fixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RawPcmEncoding {
    S16le,
    F32le,
}

impl From<RawPcmEncoding> for PcmEncoding {
    fn from(value: RawPcmEncoding) -> Self {
        match value {
            RawPcmEncoding::S16le => Self::Signed16Le,
            RawPcmEncoding::F32le => Self::Float32Le,
        }
    }
}

#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("source")
        .multiple(false)
        .args(["input", "microphone", "stdin_audio", "tcp", "unix"])
))]
pub struct FriendlyRecognitionCommand {
    /// WAV input file.
    pub input: Option<PathBuf>,
    /// Capture the local default or selected microphone.
    #[arg(long)]
    pub microphone: bool,
    /// Read headerless PCM from standard input.
    #[arg(long = "stdin")]
    pub stdin_audio: bool,
    /// Connect to a TCP source carrying headerless PCM.
    #[arg(long, value_name = "HOST:PORT")]
    pub tcp: Option<String>,
    /// Connect to a Unix-domain socket carrying headerless PCM.
    #[arg(long, value_name = "PATH")]
    pub unix: Option<PathBuf>,
    /// Exact CPAL input device ID.
    #[arg(long, requires = "microphone")]
    pub input_device: Option<String>,
    /// Headerless PCM encoding for stdin and socket inputs.
    #[arg(long, value_enum, default_value = "s16le")]
    pub pcm_encoding: RawPcmEncoding,
    /// Headerless PCM sample rate.
    #[arg(long, default_value_t = ASR_SAMPLE_RATE_HZ)]
    pub sample_rate: u32,
    /// Headerless PCM channel count.
    #[arg(long, default_value_t = 1)]
    pub channels: u16,
    /// Stop and release a live source after this many milliseconds.
    #[arg(long)]
    pub maximum_audio_ms: Option<u64>,
    /// Recognition provider.
    #[arg(long, value_enum, default_value = "whisper")]
    pub provider: RecognitionProvider,
    /// Override the installed Whisper model path.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Constrain recognition and normalization to a language tag.
    #[arg(long)]
    pub language: Option<String>,
    /// Output human text, one JSON value, or streamed JSONL events.
    #[arg(long, value_enum, default_value = "text")]
    pub output: RecognitionOutput,
    /// Show unstable partial hypotheses on stderr in text mode.
    #[arg(long)]
    pub partials: bool,
    /// Remove word/token timestamps from structured committed output.
    #[arg(long)]
    pub no_timestamps: bool,
    /// Remove speaker labels from structured committed output.
    #[arg(long)]
    pub no_speaker_labels: bool,
    /// For `listen`, write normalized mono float32 PCM to stdout.
    #[arg(long)]
    pub emit_pcm: bool,
    /// For deterministic `converse`, format the response using `{text}`.
    #[arg(long, default_value = "I heard: {text}")]
    pub response_template: String,
    /// WAV destination for the deterministic conversation response.
    #[arg(long, default_value = "conversation-response.wav")]
    pub response_wav: PathBuf,
    /// Print the library-owned workflow definition and exit.
    #[arg(long)]
    pub describe: bool,
}

enum CliAudioSource {
    Wav(WavAudioSource),
    Microphone(CpalAudioSource),
    Stdin(PcmReaderSource<std::io::Stdin>),
    Tcp(PcmReaderSource<std::net::TcpStream>),
    #[cfg(unix)]
    Unix(PcmReaderSource<std::os::unix::net::UnixStream>),
}

impl AudioSource for CliAudioSource {
    fn descriptor(&self) -> &AudioSourceDescriptor {
        match self {
            Self::Wav(source) => source.descriptor(),
            Self::Microphone(source) => source.descriptor(),
            Self::Stdin(source) => source.descriptor(),
            Self::Tcp(source) => source.descriptor(),
            #[cfg(unix)]
            Self::Unix(source) => source.descriptor(),
        }
    }

    fn next_event(&mut self) -> tongues_audio::Result<AudioSourceEvent> {
        match self {
            Self::Wav(source) => source.next_event(),
            Self::Microphone(source) => source.next_event(),
            Self::Stdin(source) => source.next_event(),
            Self::Tcp(source) => source.next_event(),
            #[cfg(unix)]
            Self::Unix(source) => source.next_event(),
        }
    }

    fn cancel(&mut self) {
        match self {
            Self::Wav(source) => source.cancel(),
            Self::Microphone(source) => source.cancel(),
            Self::Stdin(source) => source.cancel(),
            Self::Tcp(source) => source.cancel(),
            #[cfg(unix)]
            Self::Unix(source) => source.cancel(),
        }
    }
}

pub fn run(verb: FriendlySpeechVerb, command: FriendlyRecognitionCommand) -> Result<()> {
    let definition = recognition_workflow(verb);
    if command.describe {
        println!("{}", serde_json::to_string_pretty(&definition)?);
        return Ok(());
    }
    anyhow::ensure!(
        command.maximum_audio_ms != Some(0),
        "--maximum-audio-ms must be positive"
    );
    let source = NormalizedAudioSource::new(open_source(&command)?, ASR_SAMPLE_RATE_HZ, 1)?;
    if verb == FriendlySpeechVerb::Listen {
        return run_listen(source, &command);
    }
    let mut events = recognize(collect_frames(source, command.maximum_audio_ms)?, &command)?;
    apply_output_controls(&mut events, &command);
    emit_result(verb, &command, events)
}

fn apply_output_controls(events: &mut [StreamEvent], command: &FriendlyRecognitionCommand) {
    for event in events {
        if let StreamEvent::CommittedSegment {
            words, speaker_id, ..
        } = event
        {
            if command.no_timestamps {
                words.clear();
            }
            if command.no_speaker_labels {
                *speaker_id = None;
            }
        }
    }
}

fn open_source(command: &FriendlyRecognitionCommand) -> Result<CliAudioSource> {
    if let Some(input) = &command.input {
        return Ok(CliAudioSource::Wav(WavAudioSource::open(
            input,
            SOURCE_CHUNK_FRAMES,
        )?));
    }
    if command.microphone {
        eprintln!(
            "Microphone capture active; raw audio and transcript retention are disabled by default."
        );
        return Ok(CliAudioSource::Microphone(CpalAudioSource::open(
            command.input_device.as_deref(),
            32,
        )?));
    }
    let encoding = command.pcm_encoding.into();
    if command.stdin_audio {
        return Ok(CliAudioSource::Stdin(PcmReaderSource::stdin(
            encoding,
            command.sample_rate,
            command.channels,
            RAW_READ_BYTES,
        )?));
    }
    if let Some(address) = &command.tcp {
        return Ok(CliAudioSource::Tcp(PcmReaderSource::connect_tcp(
            address.as_str(),
            address.clone(),
            encoding,
            command.sample_rate,
            command.channels,
            RAW_READ_BYTES,
        )?));
    }
    #[cfg(unix)]
    if let Some(path) = &command.unix {
        return Ok(CliAudioSource::Unix(PcmReaderSource::connect_unix(
            path,
            encoding,
            command.sample_rate,
            command.channels,
            RAW_READ_BYTES,
        )?));
    }
    #[cfg(not(unix))]
    if command.unix.is_some() {
        anyhow::bail!("Unix-domain audio input is unavailable on this platform");
    }
    anyhow::bail!("an audio source is required")
}

fn collect_frames<S: AudioSource>(
    mut source: NormalizedAudioSource<S>,
    maximum_audio_ms: Option<u64>,
) -> Result<Vec<speaking::AudioFrame>> {
    let mut frames = Vec::new();
    let mut total_frames = 0_u64;
    loop {
        match source.next_event()? {
            AudioSourceEvent::Audio(chunk) => {
                total_frames = total_frames.saturating_add(chunk.audio.frames() as u64);
                frames.push(speaking::AudioFrame {
                    sample_rate_hz: chunk.audio.sample_rate_hz,
                    channels: chunk.audio.channels,
                    samples: chunk.audio.samples,
                });
                if maximum_audio_ms.is_some_and(|limit| {
                    total_frames.saturating_mul(1_000) / u64::from(ASR_SAMPLE_RATE_HZ) >= limit
                }) {
                    source.cancel();
                    break;
                }
            }
            AudioSourceEvent::Discontinuity(gap) => anyhow::bail!(
                "audio discontinuity before recognition: expected chunk {}, received {} ({})",
                gap.expected_chunk_sequence,
                gap.received_chunk_sequence,
                gap.reason
            ),
            AudioSourceEvent::EndOfStream => break,
        }
    }
    anyhow::ensure!(!frames.is_empty(), "audio source produced no samples");
    Ok(frames)
}

fn recognize(
    frames: Vec<speaking::AudioFrame>,
    command: &FriendlyRecognitionCommand,
) -> Result<Vec<StreamEvent>> {
    let mut runtime = AsrRuntime::new(AsrResourceLimits::default())?;
    let provider_id = match command.provider {
        RecognitionProvider::Fixture => {
            runtime.register_provider(FixtureAsrProvider::new(
                AsrProviderCapabilities {
                    provider_id: "fixture-cli".into(),
                    model_id: "fixture-cli-v1".into(),
                    installed: true,
                    languages: vec![LanguageId(
                        command.language.clone().unwrap_or_else(|| "en".into()),
                    )],
                    streaming: AsrStreamingCapability::Native,
                    decoding_controls: BTreeSet::from([AsrDecodingControl::Timestamps]),
                    maximum_concurrent_sessions: 1,
                    estimated_memory_mb_per_session: 1,
                    model_license: Some("fixture-only".into()),
                    model_checksum: Some("sha256:fixture".into()),
                },
                vec![
                    FixtureAsrStep {
                        text: "hello".into(),
                        confidence: Some(0.75),
                        is_final: false,
                    },
                    FixtureAsrStep {
                        text: "hello from Tongues".into(),
                        confidence: Some(0.95),
                        is_final: false,
                    },
                ],
            )?)?;
            "fixture-cli"
        }
        RecognitionProvider::Whisper => {
            let model_path = command
                .model
                .clone()
                .map(Ok)
                .unwrap_or_else(crate::models::asr_whisper_model_path)?;
            runtime.register_provider(WhisperAsrProvider::new(
                model_path,
                "whisper-large-v3-turbo",
                whisper_languages(),
            )?)?;
            "whisper.cpp"
        }
    };
    runtime.load_provider(provider_id)?;
    Ok(runtime.transcribe_offline(
        provider_id,
        AsrSessionConfig {
            language: command.language.clone().map(LanguageId),
            ..AsrSessionConfig::default()
        },
        frames,
    )?)
}

fn whisper_languages() -> Vec<LanguageId> {
    ["en", "es", "fr", "de", "it", "pt", "ja", "zh"]
        .into_iter()
        .map(|language| LanguageId(language.into()))
        .collect()
}

fn run_listen<S: AudioSource>(
    mut source: NormalizedAudioSource<S>,
    command: &FriendlyRecognitionCommand,
) -> Result<()> {
    let mut chunks = 0_u64;
    let mut total_frames = 0_u64;
    let mut stdout = std::io::stdout().lock();
    loop {
        match source.next_event()? {
            AudioSourceEvent::Audio(chunk) => {
                total_frames = total_frames.saturating_add(chunk.audio.frames() as u64);
                if command.emit_pcm {
                    for sample in chunk.audio.samples {
                        stdout.write_all(&sample.to_le_bytes())?;
                    }
                } else {
                    emit_value(
                        command.output,
                        &serde_json::json!({
                            "type": "audio_chunk",
                            "sequence": chunk.sequence,
                            "start_frame": chunk.start_frame,
                            "frames": chunk.audio.frames(),
                            "sample_rate_hz": chunk.audio.sample_rate_hz,
                            "channels": chunk.audio.channels,
                        }),
                    )?;
                }
                chunks = chunks.saturating_add(1);
                if command.maximum_audio_ms.is_some_and(|limit| {
                    total_frames.saturating_mul(1_000) / u64::from(ASR_SAMPLE_RATE_HZ) >= limit
                }) {
                    source.cancel();
                    if !command.emit_pcm {
                        emit_value(
                            command.output,
                            &serde_json::json!({
                                "type": "cancelled",
                                "reason": "maximum_audio_ms",
                                "chunks": chunks,
                            }),
                        )?;
                    }
                    break;
                }
            }
            AudioSourceEvent::Discontinuity(gap) => emit_value(
                command.output,
                &serde_json::json!({
                    "type": "discontinuity",
                    "expected_chunk_sequence": gap.expected_chunk_sequence,
                    "received_chunk_sequence": gap.received_chunk_sequence,
                    "reason": gap.reason,
                }),
            )?,
            AudioSourceEvent::EndOfStream => {
                if !command.emit_pcm {
                    emit_value(
                        command.output,
                        &serde_json::json!({"type": "completed", "chunks": chunks}),
                    )?;
                }
                break;
            }
        }
    }
    stdout.flush()?;
    Ok(())
}

fn emit_result(
    verb: FriendlySpeechVerb,
    command: &FriendlyRecognitionCommand,
    events: Vec<StreamEvent>,
) -> Result<()> {
    if command.partials && command.output == RecognitionOutput::Text {
        for event in &events {
            match event {
                StreamEvent::PartialHypothesis { text, .. } => eprintln!("~ {text}"),
                StreamEvent::RevisedHypothesis { text, .. } => eprintln!("~ [revision] {text}"),
                _ => {}
            }
        }
    }
    if verb == FriendlySpeechVerb::Transcribe {
        return match command.output {
            RecognitionOutput::Text => {
                println!("{}", committed_transcript(&events));
                Ok(())
            }
            RecognitionOutput::Json => {
                println!("{}", serde_json::to_string_pretty(&events)?);
                Ok(())
            }
            RecognitionOutput::Jsonl => emit_events_jsonl(&events),
        };
    }

    let mut pipeline = CommittedTranscriptPipeline::new(
        RuleBasedTranscriptNormalizer::default(),
        StructuralTranscriptInterpreter,
    );
    let mut artifacts = Vec::new();
    for event in &events {
        if let Some(output) = pipeline.process_event(
            event,
            TranscriptSourceMetadata {
                event_ref: None,
                event_times: None,
                provenance: None,
            },
        )? {
            artifacts.push(output);
        }
    }
    if verb == FriendlySpeechVerb::Converse {
        let transcript = artifacts
            .iter()
            .map(|output| output.transcript.downstream_text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let response = command.response_template.replace("{text}", &transcript);
        write_mock_response_wav(&response, &command.response_wav)?;
        return emit_value(
            command.output,
            &serde_json::json!({
                "workflow": recognition_workflow(verb),
                "transcript": transcript,
                "response": response,
                "audio_output": command.response_wav,
                "tts_backend": "deterministic_mock",
            }),
        );
    }

    match command.output {
        RecognitionOutput::Text | RecognitionOutput::Json => {
            let values = artifacts
                .into_iter()
                .map(|output| {
                    if verb == FriendlySpeechVerb::Recognize {
                        serde_json::json!({
                            "transcript": output.transcript,
                            "syntax": output.syntax,
                        })
                    } else {
                        serde_json::to_value(output).expect("artifact is serializable")
                    }
                })
                .collect::<Vec<_>>();
            println!("{}", serde_json::to_string_pretty(&values)?);
            Ok(())
        }
        RecognitionOutput::Jsonl => {
            for output in artifacts {
                if verb == FriendlySpeechVerb::Recognize {
                    println!(
                        "{}",
                        serde_json::json!({
                            "transcript": output.transcript,
                            "syntax": output.syntax,
                        })
                    );
                } else {
                    println!("{}", serde_json::to_string(&output)?);
                }
            }
            Ok(())
        }
    }
}

fn write_mock_response_wav(text: &str, path: &PathBuf) -> Result<()> {
    let plan = tongues_tts::utterance_plan_from_text(tongues_tts::SpeechRequest {
        text: text.into(),
        variety: "en-US-GA".into(),
    })?;
    let renderer = tongues_tts::MockTtsRenderer::new(Default::default());
    let pcm = renderer.synthesize_plan_to_vec(&plan);
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 22_050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .with_context(|| format!("creating {}", path.display()))?;
    for sample in pcm {
        writer.write_sample((sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn emit_events_jsonl(events: &[StreamEvent]) -> Result<()> {
    for event in events {
        println!("{}", serde_json::to_string(event)?);
    }
    Ok(())
}

fn emit_value(output: RecognitionOutput, value: &serde_json::Value) -> Result<()> {
    match output {
        RecognitionOutput::Text => println!(
            "{}",
            value
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("result")
        ),
        RecognitionOutput::Json => println!("{}", serde_json::to_string_pretty(value)?),
        RecognitionOutput::Jsonl => println!("{}", serde_json::to_string(value)?),
    }
    Ok(())
}
