use anyhow::{Context, Result};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::backend::Backend;
use burn_cuda::{Cuda, CudaDevice};
use clap::Args;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::DeviceArg;
use speaking::{
    phone_display_symbol, phoneme_default_phone_display_symbol, phonemicizer_for_variety,
    EvidenceProvenance, EvidenceSource, FeatureId, FeatureValue, PauseKind, PhonemicizeOutput,
    PhonemicizeRequest, PronunciationWarning, PronunciationWarningKind, ProsodyTrack, SpeakerId,
    SpeakerReference, SpeakerReferenceSource, Spec, SpeechBoundaryToken, StyleRef, StyleSource,
    TerminalPunctuation, UtteranceId, UtterancePlan, VarietyId,
};
use speech::LinguisticProjector as _;
use styletts2::{
    prepare_styletts2_plan, styletts2_en_us_symbol_set, styletts2_text_for_symbols,
    validate_styletts2_plan, MockStyleTts2Backend, StyleTts2Backend, StyleTts2PlanOptions,
    StyleTts2SynthesisRequest, StyleTts2Timing, DEFAULT_MAX_TTS_SYMBOLS,
};
use tongues_tts as speech;

#[cfg(feature = "styletts2-onnx")]
use styletts2::{StyleTts2DiffusionOptions, StyleTts2OnnxBackend};

const DEFAULT_STYLE_ALPHA: f32 = 0.3;
const DEFAULT_STYLE_BETA: f32 = 0.1;
const DEFAULT_SPEED: f64 = 1.0;

#[derive(Debug, Args, Clone)]
pub struct SpeakCommand {
    #[arg(help = "The text to speak. If not provided, reads from stdin.")]
    pub text: Option<String>,
    #[arg(
        long,
        default_value = "en-US",
        help = "Language or pronunciation variety tag"
    )]
    pub variety: String,
    #[arg(long, value_enum, default_value_t = SpeakBackend::Burn, help = "Speech backend to use")]
    pub backend: SpeakBackend,
    #[arg(
        long,
        short,
        help = "WAV output path. If omitted, writes speech audio to stdout where supported."
    )]
    pub output: Option<PathBuf>,
    #[arg(long, default_value_t = 24_000, help = "Output sample rate in Hz")]
    pub sample_rate_hz: u32,
    #[arg(
        long,
        help = "Named speaker from the selected voice model, such as p225"
    )]
    pub speaker: Option<String>,
    #[arg(long, help = "Numeric speaker ID for low-level voice model testing")]
    pub speaker_id: Option<u32>,
    #[arg(long, help = "List speakers declared by the selected voice model")]
    pub list_speakers: bool,
    #[arg(long, help = "Reference WAV for speaker timbre")]
    pub voice_wav: Option<PathBuf>,
    #[arg(long, help = "Reference WAV for style and prosody")]
    pub style_wav: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = SpeakQuality::Balanced, help = "Quality preset for backend defaults")]
    pub quality: SpeakQuality,
    #[arg(long, help = "Override diffusion steps from the quality preset")]
    pub diffusion_steps: Option<usize>,
    #[arg(
        long,
        help = "Reference voice strength in 0..=1; higher keeps more speaker timbre from --voice-wav"
    )]
    pub speaker_reference_strength: Option<f32>,
    #[arg(
        long,
        help = "Reference style strength in 0..=1; higher keeps more style/prosody from --style-wav"
    )]
    pub style_reference_strength: Option<f32>,
    #[arg(
        long,
        default_value_t = DEFAULT_STYLE_ALPHA,
        help = "Raw StyleTTS2 alpha blend; higher uses more predicted speaker/timbre and less reference"
    )]
    pub style_alpha: f32,
    #[arg(
        long,
        default_value_t = DEFAULT_STYLE_BETA,
        help = "Raw StyleTTS2 beta blend; higher uses more predicted style/prosody and less reference"
    )]
    pub style_beta: f32,
    #[arg(
        long,
        help = "Path to a JSON file containing emotion signatures (deltas)"
    )]
    pub emotion_signatures: Option<PathBuf>,
    #[arg(
        long,
        help = "Name of the target emotion to apply (requires --emotion-signatures)"
    )]
    pub emotion: Option<String>,
    #[arg(
        long,
        default_value_t = 1.0,
        help = "Multiplier for the emotion delta vector"
    )]
    pub emotion_strength: f32,
    #[arg(
        long,
        default_value_t = 1.0,
        help = "StyleTTS2 diffusion embedding scale"
    )]
    pub embedding_scale: f64,
    #[arg(long, default_value_t = 0, help = "Seed for StyleTTS2 style diffusion")]
    pub style_seed: u64,
    #[arg(long, default_value_t = DEFAULT_SPEED, help = "StyleTTS2 decoder speed multiplier")]
    pub speed: f64,
    #[arg(long, help = "Latent noise scale for stochastic speech backends")]
    pub noise_scale: Option<f32>,
    #[arg(long, help = "Duration noise scale for stochastic speech backends")]
    pub duration_noise_scale: Option<f32>,
    #[arg(long, help = "RNG seed for repeatable stochastic speech inference")]
    pub seed: Option<u64>,
    #[arg(long, help = "Print pronunciation planning diagnostics")]
    pub debug_pronunciation: bool,
    #[arg(long, help = "Emit word and audio timing metadata")]
    pub timings: bool,
    #[arg(
        long,
        default_value_t = 1,
        help = "Synthesize the same planned input repeatedly in one process; run 1 is cold and later runs are warm"
    )]
    pub benchmark_runs: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_TTS_SYMBOLS, help = "Maximum symbols per TTS chunk")]
    pub max_tts_symbols: usize,
    #[arg(long, help = "Disable automatic text chunking before TTS")]
    pub no_tts_chunking: bool,
    #[arg(long, help = "Exit with an error when pronunciation must be guessed")]
    pub fail_on_guessed_pronunciation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SpeakBackend {
    Burn,
    Vits,
    Mock,
    Styletts2,
    Onnx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SpeakQuality {
    Balanced,
    Fast,
}

impl SpeakQuality {
    pub fn diffusion_steps(self) -> usize {
        match self {
            Self::Balanced => 5,
            Self::Fast => 2,
        }
    }
}

impl SpeakCommand {
    pub fn resolved_diffusion_steps(&self) -> usize {
        self.diffusion_steps
            .unwrap_or_else(|| self.quality.diffusion_steps())
    }

    pub fn resolved_style_alpha(&self) -> f32 {
        self.speaker_reference_strength
            .map(reference_strength_to_blend)
            .unwrap_or(self.style_alpha)
    }

    pub fn resolved_style_beta(&self) -> f32 {
        self.style_reference_strength
            .map(reference_strength_to_blend)
            .unwrap_or(self.style_beta)
    }
}

fn reference_strength_to_blend(strength: f32) -> f32 {
    1.0 - strength
}

#[cfg(feature = "onnx-tts")]
fn onnx_synthesis_options(options: &SpeechSynthesisOptions) -> Result<speech::SynthesisOptions> {
    if options.speaker.is_some() && options.speaker_id.is_some() {
        anyhow::bail!("use either --speaker or --speaker-id, not both");
    }
    Ok(speech::SynthesisOptions {
        speaker_id: options.speaker_id,
        split_sentences: true,
        length_scale: None,
        noise_scale: None,
        noise_w: None,
        seed: None,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSynthesisArtifact {
    pub sample_rate_hz: u32,
    pub pcm: Vec<f32>,
    pub timings: Vec<StyleTts2Timing>,
    pub profile: Vec<speech::SynthesisProfileEvent>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSynthesisOptions {
    pub sample_rate_hz: u32,
    pub speaker: Option<String>,
    pub speaker_id: Option<u32>,
    pub voice_wav: Option<PathBuf>,
    pub style_wav: Option<PathBuf>,
    pub diffusion_steps: usize,
    pub style_alpha: f32,
    pub style_beta: f32,
    pub embedding_scale: f64,
    pub style_seed: u64,
    pub speed: f64,
    pub noise_scale: Option<f32>,
    pub duration_noise_scale: Option<f32>,
    pub seed: Option<u64>,
    pub max_tts_symbols: usize,
    pub no_tts_chunking: bool,
    pub emotion_signatures: Option<PathBuf>,
    pub emotion: Option<String>,
    pub emotion_strength: f32,
}

impl From<&SpeakCommand> for SpeechSynthesisOptions {
    fn from(command: &SpeakCommand) -> Self {
        Self {
            sample_rate_hz: command.sample_rate_hz,
            speaker: command.speaker.clone(),
            speaker_id: command.speaker_id,
            voice_wav: command.voice_wav.clone(),
            style_wav: command.style_wav.clone(),
            diffusion_steps: command.resolved_diffusion_steps(),
            style_alpha: command.resolved_style_alpha(),
            style_beta: command.resolved_style_beta(),
            embedding_scale: command.embedding_scale,
            style_seed: command.style_seed,
            speed: command.speed,
            noise_scale: command.noise_scale,
            duration_noise_scale: command.duration_noise_scale,
            seed: command.seed,
            max_tts_symbols: command.max_tts_symbols,
            no_tts_chunking: command.no_tts_chunking,
            emotion_signatures: command.emotion_signatures.clone(),
            emotion: command.emotion.clone(),
            emotion_strength: command.emotion_strength,
        }
    }
}

type CpuBurnBackend = speech::BurnSpeedySpeechPipeline<NdArray<f32>>;
type CudaBurnBackend = speech::BurnSpeedySpeechPipeline<Cuda<f32, i32>>;
type CpuVitsBackend = speech::BurnVitsSpeech<NdArray<f32>>;
type CudaVitsBackend = speech::BurnVitsSpeech<Cuda<f32, i32>>;
type AudioCallback<'a> = Option<&'a mut dyn FnMut(&[f32])>;

#[allow(dead_code)]
enum BackendInstance {
    BurnCpu(Box<CpuBurnBackend>),
    BurnCuda(Box<CudaBurnBackend>),
    VitsCpu(Box<CpuVitsBackend>),
    VitsCuda(Box<CudaVitsBackend>),
    Mock(MockStyleTts2Backend),
    #[cfg(feature = "styletts2-onnx")]
    StyleTts2(StyleTts2OnnxBackend),
    #[cfg(not(feature = "styletts2-onnx"))]
    StyleTts2,
    #[cfg(feature = "onnx-tts")]
    Onnx(speech::OnnxSpeechBackend),
    #[cfg(not(feature = "onnx-tts"))]
    Onnx,
}

impl BackendInstance {
    fn label(&self) -> &'static str {
        match self {
            Self::BurnCpu(_) => "burn-cpu",
            Self::BurnCuda(_) => "burn-cuda",
            Self::VitsCpu(_) => "vits-cpu",
            Self::VitsCuda(_) => "vits-cuda",
            Self::Mock(_) => "mock",
            Self::StyleTts2 { .. } => "styletts2",
            Self::Onnx { .. } => "onnx",
        }
    }

    fn synthesize(
        &mut self,
        plan: &UtterancePlan,
        options: &SpeechSynthesisOptions,
        mut on_audio: AudioCallback<'_>,
        command: &SpeakCommand,
    ) -> Result<SpeechSynthesisArtifact> {
        #[cfg(not(any(feature = "styletts2-onnx", feature = "onnx-tts")))]
        let _ = options;

        match self {
            Self::BurnCpu(ref mut backend) => {
                synthesize_burn_engine(backend.as_mut(), plan, options, on_audio, command.timings)
            }
            Self::BurnCuda(ref mut backend) => {
                synthesize_burn_engine(backend.as_mut(), plan, options, on_audio, command.timings)
            }
            Self::VitsCpu(ref mut backend) => {
                synthesize_burn_engine(backend.as_mut(), plan, options, on_audio, command.timings)
            }
            Self::VitsCuda(ref mut backend) => {
                synthesize_burn_engine(backend.as_mut(), plan, options, on_audio, command.timings)
            }
            Self::Mock(ref mut backend) => {
                let styletts2_plan = prepare_styletts2_plan(
                    plan,
                    &styletts2_en_us_symbol_set(),
                    styletts2_options_from(command.max_tts_symbols, command.no_tts_chunking),
                )
                .context("failed to prepare StyleTTS2 synthesis plan")?;
                validate_styletts2_plan(&styletts2_plan)
                    .context("invalid StyleTTS2 synthesis plan")?;
                let request = StyleTts2SynthesisRequest::from_backend_plan(
                    styletts2_plan,
                    None,
                    None,
                    ProsodyTrack::default(),
                );
                let mut pcm_mono_f32 = Vec::new();
                let output = backend
                    .synthesize_streaming(&request, &mut |chunk: styletts2::StyleTts2AudioChunk| {
                        pcm_mono_f32.extend(&chunk.pcm_mono_f32);
                        if let Some(ref mut cb) = on_audio {
                            cb(&chunk.pcm_mono_f32);
                        }
                        Ok(())
                    })
                    .context("mock StyleTTS2 synthesis failed")?;

                Ok(SpeechSynthesisArtifact {
                    sample_rate_hz: output.sample_rate_hz,
                    pcm: pcm_mono_f32,
                    timings: output.timings,
                    profile: Vec::new(),
                })
            }
            #[cfg(feature = "styletts2-onnx")]
            Self::StyleTts2(ref mut backend) => {
                let styletts2_plan = prepare_styletts2_plan(
                    plan,
                    &styletts2_en_us_symbol_set(),
                    styletts2_options_from(command.max_tts_symbols, command.no_tts_chunking),
                )
                .context("failed to prepare StyleTTS2 synthesis plan")?;
                let default_refs =
                    crate::models::ensure_styletts2_default_reference_audio_available()?;
                let voice_ref = options.voice_wav.as_ref().unwrap_or(&default_refs.voice);
                let style_ref = options.style_wav.as_ref().unwrap_or(&default_refs.style);

                let mut style_ref_clone = plan.style.clone();
                let style_uri = style_ref.display().to_string();

                if let (Some(signatures_path), Some(emotion_name)) =
                    (&options.emotion_signatures, &options.emotion)
                {
                    let mut base_style_vector = backend
                        .reference_style_vector(&format!("file://{}", style_uri))
                        .context("Failed to get base style vector")?;

                    let sig_file = std::fs::File::open(signatures_path)
                        .context("Failed to open emotion signatures")?;
                    let sigs: serde_json::Value = serde_json::from_reader(sig_file)
                        .context("Failed to parse emotion signatures")?;

                    if let Some(sig) = sigs.get(emotion_name) {
                        if let Some(delta_vec) = sig.get("vector").and_then(|v| v.as_array()) {
                            let delta: Vec<f32> = delta_vec
                                .iter()
                                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                                .collect();
                            for i in 0..256 {
                                base_style_vector[i] += delta[i] * options.emotion_strength;
                            }

                            style_ref_clone = Some(speaking::StyleRef {
                                description: None,
                                source: speaking::StyleSource::Embedding {
                                    kind: "styletts2.emotion.v1".into(),
                                    values: base_style_vector,
                                },
                            });
                        }
                    } else {
                        anyhow::bail!("Emotion {} not found in signatures file", emotion_name);
                    }
                }

                let request = StyleTts2SynthesisRequest::from_backend_plan(
                    styletts2_plan,
                    plan.speaker.clone(),
                    style_ref_clone,
                    plan.target_prosody.clone(),
                )
                .with_speaker_reference_audio_uri(voice_ref.display().to_string())
                .with_style_reference_audio_uri(style_uri);

                let mut pcm_mono_f32 = Vec::new();
                let output = backend
                    .synthesize_streaming(&request, &mut |chunk: styletts2::StyleTts2AudioChunk| {
                        pcm_mono_f32.extend(&chunk.pcm_mono_f32);
                        if let Some(ref mut cb) = on_audio {
                            cb(&chunk.pcm_mono_f32);
                        }
                        Ok(())
                    })
                    .context("native StyleTTS2 ONNX synthesis failed")?;

                Ok(SpeechSynthesisArtifact {
                    sample_rate_hz: output.sample_rate_hz,
                    pcm: pcm_mono_f32,
                    timings: output.timings,
                    profile: Vec::new(),
                })
            }
            #[cfg(not(feature = "styletts2-onnx"))]
            Self::StyleTts2 => {
                anyhow::bail!(
                    "StyleTTS2 native backend requires compiling with feature `styletts2-onnx`"
                )
            }
            #[cfg(feature = "onnx-tts")]
            Self::Onnx(ref mut backend) => {
                let mut pcm_mono_f32 = Vec::new();
                let synthesis_options = onnx_synthesis_options(options)?;
                backend.synthesize_plan_streaming_with_options(
                    plan,
                    &synthesis_options,
                    &mut |chunk: speech::AudioChunk| {
                        pcm_mono_f32.extend(&chunk.pcm_mono_f32);
                        if let Some(ref mut cb) = on_audio {
                            cb(&chunk.pcm_mono_f32);
                        }
                        Ok(())
                    },
                )?;

                Ok(SpeechSynthesisArtifact {
                    sample_rate_hz: backend.sample_rate_hz(),
                    pcm: pcm_mono_f32,
                    timings: Vec::new(),
                    profile: Vec::new(),
                })
            }
            #[cfg(not(feature = "onnx-tts"))]
            Self::Onnx => {
                anyhow::bail!("ONNX speech backend requires compiling with feature `onnx-tts`")
            }
        }
    }

    fn projected_input(&self, plan: &UtterancePlan) -> Result<Option<speech::PhonemeTokenIds>> {
        match self {
            Self::BurnCpu(backend) => Ok(Some(
                backend
                    .acoustic_model()
                    .projector()
                    .project(plan)
                    .context("failed to project SpeedySpeech checkpoint input")?,
            )),
            Self::BurnCuda(backend) => Ok(Some(
                backend
                    .acoustic_model()
                    .projector()
                    .project(plan)
                    .context("failed to project SpeedySpeech checkpoint input")?,
            )),
            Self::VitsCpu(backend) => Ok(Some(
                backend
                    .projected_input(plan)
                    .context("failed to project VITS checkpoint input")?,
            )),
            Self::VitsCuda(backend) => Ok(Some(
                backend
                    .projected_input(plan)
                    .context("failed to project VITS checkpoint input")?,
            )),
            Self::Mock(_) | Self::StyleTts2 { .. } | Self::Onnx { .. } => Ok(None),
        }
    }
}

fn synthesize_burn_engine(
    backend: &mut dyn speech::SpeechSynthesisEngine,
    plan: &UtterancePlan,
    options: &SpeechSynthesisOptions,
    mut on_audio: AudioCallback<'_>,
    profiling: bool,
) -> Result<SpeechSynthesisArtifact> {
    anyhow::ensure!(
        options.speed.is_finite() && options.speed > 0.0,
        "--speed must be finite and positive"
    );
    let request = speech::SpeechSynthesisRequest {
        plan: plan.clone(),
        options: speech::SynthesisOptions {
            speaker_id: options.speaker_id,
            split_sentences: true,
            length_scale: Some((1.0 / options.speed) as f32),
            noise_scale: options.noise_scale,
            noise_w: options.duration_noise_scale,
            seed: options.seed,
        },
    };
    let sample_rate_hz = backend.sample_rate_hz();
    let mut pcm = Vec::new();
    let mut profile = Vec::new();
    let mut sink = |chunk: speech::AudioChunk| {
        pcm.extend(&chunk.pcm_mono_f32);
        if let Some(ref mut cb) = on_audio {
            cb(&chunk.pcm_mono_f32);
        }
        Ok(())
    };
    if profiling {
        backend.synthesize_plan_streaming_profiled(&request, &mut sink, &mut |event| {
            profile.push(event)
        })?;
    } else {
        backend.synthesize_plan_streaming(&request, &mut sink)?;
    }
    Ok(SpeechSynthesisArtifact {
        sample_rate_hz,
        pcm,
        timings: Vec::new(),
        profile,
    })
}

fn split_into_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        current.push(c);
        if c == '.' || c == '?' || c == '!' {
            let is_boundary = if i + 1 < chars.len() {
                chars[i + 1].is_whitespace()
            } else {
                true
            };
            if is_boundary {
                let s = current.trim().to_string();
                if !s.is_empty() {
                    sentences.push(s);
                }
                current.clear();
            }
        } else if current.len() >= 150 && c.is_whitespace() {
            let s = current.trim().to_string();
            if !s.is_empty() {
                sentences.push(s);
            }
            current.clear();
        }
        i += 1;
    }
    let s = current.trim().to_string();
    if !s.is_empty() {
        sentences.push(s);
    }
    sentences
}

pub fn run_speak(command: SpeakCommand, device_arg: DeviceArg) -> Result<()> {
    anyhow::ensure!(
        (1..=32).contains(&command.benchmark_runs),
        "--benchmark-runs must be between 1 and 32"
    );
    let options = SpeechSynthesisOptions::from(&command);

    if command.list_speakers {
        print_available_speakers(&command)?;
        return Ok(());
    }

    let target_sample_rate = match command.backend {
        SpeakBackend::Burn => 22_050,
        SpeakBackend::Vits => 22_050,
        SpeakBackend::Mock => command.sample_rate_hz,
        SpeakBackend::Styletts2 => command.sample_rate_hz,
        SpeakBackend::Onnx => 22050,
    };

    let startup_started = Instant::now();
    let mut cache_check_ms = 0.0;
    let mut model_load_ms = 0.0;
    let mut model_load_profile = Vec::new();
    let mut backend = match command.backend {
        SpeakBackend::Burn => {
            let started = Instant::now();
            let acoustic_checkpoint =
                crate::models::ensure_model_available(crate::models::DEFAULT_ACOUSTIC_MODEL_ID)?;
            let vocoder_checkpoint =
                crate::models::ensure_model_available(crate::models::DEFAULT_NEURAL_VOCODER_ID)?;
            cache_check_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let acoustic_config = component_config_path(&acoustic_checkpoint)?;
            let vocoder_config = component_config_path(&vocoder_checkpoint)?;
            let started = Instant::now();
            let backend = match device_arg {
                DeviceArg::Cpu => {
                    BackendInstance::BurnCpu(Box::new(load_burn_pipeline::<NdArray<f32>>(
                        &acoustic_config,
                        &acoustic_checkpoint,
                        &vocoder_config,
                        &vocoder_checkpoint,
                        NdArrayDevice::Cpu,
                        &mut |event| model_load_profile.push(event),
                    )?))
                }
                DeviceArg::Cuda => {
                    BackendInstance::BurnCuda(Box::new(load_burn_pipeline::<Cuda<f32, i32>>(
                        &acoustic_config,
                        &acoustic_checkpoint,
                        &vocoder_config,
                        &vocoder_checkpoint,
                        CudaDevice::default(),
                        &mut |event| model_load_profile.push(event),
                    )?))
                }
            };
            model_load_ms = started.elapsed().as_secs_f64() * 1_000.0;
            backend
        }
        SpeakBackend::Vits => {
            let started = Instant::now();
            let checkpoint = crate::models::ensure_model_available(
                crate::models::DEFAULT_END_TO_END_SPEECH_MODEL_ID,
            )?;
            cache_check_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let config = component_config_path(&checkpoint)?;
            let speakers = checkpoint
                .parent()
                .context("VITS checkpoint path has no parent directory")?
                .join("speaker_ids.json");
            let started = Instant::now();
            let backend = match device_arg {
                DeviceArg::Cpu => BackendInstance::VitsCpu(Box::new(
                    speech::BurnVitsSpeech::load_profiled(
                        &config,
                        &checkpoint,
                        &speakers,
                        NdArrayDevice::Cpu,
                        &mut |event| model_load_profile.push(event),
                    )
                    .context("failed to load Burn VITS speech model on CPU")?,
                )),
                DeviceArg::Cuda => BackendInstance::VitsCuda(Box::new(
                    speech::BurnVitsSpeech::load_profiled(
                        &config,
                        &checkpoint,
                        &speakers,
                        CudaDevice::default(),
                        &mut |event| model_load_profile.push(event),
                    )
                    .context("failed to load Burn VITS speech model on CUDA")?,
                )),
            };
            model_load_ms = started.elapsed().as_secs_f64() * 1_000.0;
            backend
        }
        SpeakBackend::Mock => {
            BackendInstance::Mock(MockStyleTts2Backend::new(command.sample_rate_hz))
        }
        SpeakBackend::Styletts2 => {
            #[cfg(feature = "styletts2-onnx")]
            {
                let primary_model = crate::models::ensure_styletts2_model_available()?;
                let model_dir = primary_model
                    .parent()
                    .context("StyleTTS2 primary model path has no parent directory")?;

                let diffusion_opts = StyleTts2DiffusionOptions {
                    diffusion_steps: options.diffusion_steps,
                    alpha: options.style_alpha,
                    beta: options.style_beta,
                    embedding_scale: options.embedding_scale,
                    seed: options.style_seed,
                };

                let backend = StyleTts2OnnxBackend::from_model_dir(model_dir)
                    .context("failed to load native StyleTTS2 ONNX backend")?
                    .with_diffusion_options(diffusion_opts)
                    .context("invalid StyleTTS2 diffusion options")?
                    .with_speed(options.speed)
                    .context("invalid StyleTTS2 speed")?;
                BackendInstance::StyleTts2(backend)
            }
            #[cfg(not(feature = "styletts2-onnx"))]
            {
                anyhow::bail!(
                    "StyleTTS2 native backend requires compiling with feature `styletts2-onnx`"
                )
            }
        }
        SpeakBackend::Onnx => {
            #[cfg(feature = "onnx-tts")]
            {
                use speech::{voice_config_path, OnnxSpeechBackend, VoiceConfig};
                let primary_model = crate::models::ensure_voice_model_available()?;
                let config_path = voice_config_path(&primary_model);
                let config = VoiceConfig::from_json_file(&config_path)?;
                let backend = match device_arg {
                    DeviceArg::Cpu => OnnxSpeechBackend::load_cpu(&primary_model, config)?,
                    DeviceArg::Cuda => OnnxSpeechBackend::load(&primary_model, config)?,
                };
                BackendInstance::Onnx(backend)
            }
            #[cfg(not(feature = "onnx-tts"))]
            {
                anyhow::bail!("ONNX speech backend requires compiling with feature `onnx-tts`")
            }
        }
    };
    let backend_label = backend.label();
    let cold_start_ms = startup_started.elapsed().as_secs_f64() * 1_000.0;
    if command.timings {
        println!(
            "startup_profile_json: {}",
            serde_json::json!({
                "backend": backend_label,
                "download_cache_check_ms": cache_check_ms,
                "config_model_construction_weight_upload_ms": model_load_ms,
                "model_load_stages": model_load_profile,
                "total_cold_start_ms": cold_start_ms,
            })
        );
    }

    let player = if command.output.is_none() {
        match AudioStreamPlayer::new(target_sample_rate) {
            Ok(p) => Some(p),
            Err(e) => {
                println!(
                    "Warning: Failed to initialize audio player: {}. Playing audio will be skipped.",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    let mut all_pcm = Vec::new();
    let mut all_timings = Vec::new();
    let mut total_samples = 0;

    let process_chunk = |text_chunk: &str,
                         backend: &mut BackendInstance,
                         player: &Option<AudioStreamPlayer>,
                         all_pcm: &mut Vec<f32>,
                         all_timings: &mut Vec<StyleTts2Timing>,
                         total_samples: &mut usize|
     -> Result<()> {
        if text_chunk.trim().is_empty() {
            return Ok(());
        }

        let planning_started = Instant::now();
        let variety = VarietyId(command.variety.clone());
        let phonemicizer = phonemicizer_for_variety(&variety)
            .map_err(|e| anyhow::anyhow!("failed to load phonemicizer: {e}"))?;
        let phonemicized = phonemicizer
            .phonemicize(&PhonemicizeRequest {
                text: text_chunk.to_string(),
                variety,
                style: None,
            })
            .context("failed to phonemicize text into a speech plan")?;

        let mut plan = utterance_plan_from_phonemicized(&phonemicized);
        plan.speaker = options.speaker.clone().map(SpeakerId);
        plan.speaker_reference = options.voice_wav.as_ref().map(|path| SpeakerReference {
            description: Some("CLI speaker reference".into()),
            source: SpeakerReferenceSource::ReferenceAudio {
                uri: path.display().to_string(),
            },
        });
        if let Some(path) = options.style_wav.as_ref() {
            plan.style = Some(StyleRef {
                description: Some("CLI style reference".into()),
                source: StyleSource::ReferenceAudio {
                    uri: path.display().to_string(),
                },
            });
        }
        let planning_ms = planning_started.elapsed().as_secs_f64() * 1_000.0;

        if plan.intended_phonemes.is_empty() {
            return Ok(());
        }

        if command.fail_on_guessed_pronunciation
            && phonemicized.warnings.iter().any(is_guessed_pronunciation)
        {
            anyhow::bail!(
                "guessed pronunciation encountered: {}",
                phonemicized
                    .warnings
                    .iter()
                    .filter(|warning| is_guessed_pronunciation(warning))
                    .map(|warning| warning.token.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        let diagnostic_projection_started = Instant::now();
        let checkpoint_input = backend.projected_input(&plan)?;
        let diagnostic_projection_ms =
            diagnostic_projection_started.elapsed().as_secs_f64() * 1_000.0;
        let mut output_artifact = None;
        for run_index in 0..command.benchmark_runs {
            let synthesis_started = Instant::now();
            let mut first_audio_latency_ms = None;
            let mut audio_callback = |chunk: &[f32]| {
                if first_audio_latency_ms.is_none() {
                    first_audio_latency_ms =
                        Some(synthesis_started.elapsed().as_secs_f64() * 1_000.0);
                }
                if run_index == 0 {
                    if let Some(ref p) = player {
                        p.append(chunk);
                    }
                }
            };
            let artifact =
                backend.synthesize(&plan, &options, Some(&mut audio_callback), &command)?;
            let elapsed_ms = synthesis_started.elapsed().as_secs_f64() * 1_000.0;
            let audio_seconds = artifact.pcm.len() as f64 / artifact.sample_rate_hz as f64;
            let real_time_factor = elapsed_ms / 1_000.0 / audio_seconds;
            let cold_end_to_end_ms = (run_index == 0)
                .then_some(cold_start_ms + planning_ms + diagnostic_projection_ms + elapsed_ms);
            if command.timings {
                println!(
                    "inference_profile_json: {}",
                    serde_json::json!({
                        "run": run_index + 1,
                        "temperature": if run_index == 0 { "cold" } else { "warm" },
                        "planning_ms": planning_ms,
                        "diagnostic_checkpoint_projection_ms": diagnostic_projection_ms,
                        "first_playable_audio_latency_ms": first_audio_latency_ms,
                        "total_synthesis_ms": elapsed_ms,
                        "audio_seconds": audio_seconds,
                        "real_time_factor": real_time_factor,
                        "cold_end_to_end_ms": cold_end_to_end_ms,
                        "cold_end_to_end_real_time_factor": cold_end_to_end_ms
                            .map(|total_ms| total_ms / 1_000.0 / audio_seconds),
                        "stages": &artifact.profile,
                    })
                );
            }
            if run_index == 0 {
                output_artifact = Some(artifact);
            }
        }
        let artifact = output_artifact.context("synthesis produced no benchmark runs")?;

        let backend_symbols = match (&checkpoint_input, command.backend) {
            (Some(projected), SpeakBackend::Burn | SpeakBackend::Vits) => {
                projected.projected_symbols.clone()
            }
            (None, SpeakBackend::Burn | SpeakBackend::Vits) => {
                anyhow::bail!("native Burn backend did not expose its checkpoint projection")
            }
            (_, SpeakBackend::Mock | SpeakBackend::Styletts2) => {
                let styletts2_plan = prepare_styletts2_plan(
                    &plan,
                    &styletts2_en_us_symbol_set(),
                    styletts2_options_from(command.max_tts_symbols, command.no_tts_chunking),
                )
                .context("failed to prepare StyleTTS2 synthesis plan")?;
                styletts2_plan
                    .chunks
                    .iter()
                    .map(|chunk| {
                        styletts2_text_for_symbols(&chunk.symbols)
                            .map(|text| text.trim().to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .context("failed to format StyleTTS2 backend symbols")?
                    .join(" || ")
            }
            (_, SpeakBackend::Onnx) => {
                let sequence = speech::phoneme_sequence_from_plan(&plan)?;
                sequence.symbols.join(" ")
            }
        };

        println!("Tongues speech synthesis plan");
        println!("backend: {backend_label}");
        println!("variety: {}", phonemicized.variety.0);
        println!("text: {}", phonemicized.text);
        println!("phonemes: {}", format_phonemes(&phonemicized));
        if command.debug_pronunciation {
            println!(
                "phonemes_debug: {}",
                format_phonemes_with_features(&phonemicized)
            );
        }
        println!("phones: {}", format_phones(&phonemicized));
        println!("checkpoint_symbols: {backend_symbols}");
        if let Some(projected) = &checkpoint_input {
            println!(
                "checkpoint_token_ids: {}",
                projected
                    .ids
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        if matches!(command.backend, SpeakBackend::Vits) {
            println!(
                "inference_seed: {}",
                options
                    .seed
                    .map(|seed| seed.to_string())
                    .unwrap_or_else(|| "<random>".into())
            );
            println!(
                "latent_noise_scale: {}",
                options
                    .noise_scale
                    .map(|scale| scale.to_string())
                    .unwrap_or_else(|| "<checkpoint default>".into())
            );
            println!(
                "duration_noise_scale: {}",
                options
                    .duration_noise_scale
                    .map(|scale| scale.to_string())
                    .unwrap_or_else(|| "<checkpoint default>".into())
            );
        }

        if matches!(command.backend, SpeakBackend::Styletts2) {
            println!("styletts2_controls:");
            println!("  diffusion_steps: {}", options.diffusion_steps);
            println!(
                "  speaker_reference_strength: {:.3}",
                1.0 - options.style_alpha
            );
            println!(
                "  style_reference_strength: {:.3}",
                1.0 - options.style_beta
            );
            println!("  alpha: {:.3}", options.style_alpha);
            println!("  beta: {:.3}", options.style_beta);
            println!("  embedding_scale: {:.3}", options.embedding_scale);
            println!("  style_seed: {}", options.style_seed);
            println!("  speed: {:.3}", options.speed);
        }

        println!("chunks:");
        match command.backend {
            SpeakBackend::Burn | SpeakBackend::Vits => {
                println!("  1: {backend_symbols}");
            }
            SpeakBackend::Mock | SpeakBackend::Styletts2 => {
                let styletts2_plan = prepare_styletts2_plan(
                    &plan,
                    &styletts2_en_us_symbol_set(),
                    styletts2_options_from(command.max_tts_symbols, command.no_tts_chunking),
                )
                .context("failed to prepare StyleTTS2 synthesis plan")?;
                for (index, chunk) in styletts2_plan.chunks.iter().enumerate() {
                    println!(
                        "  {}: {}",
                        index + 1,
                        styletts2_text_for_symbols(&chunk.symbols)
                            .map(|text| text.trim().to_string())
                            .context("failed to format StyleTTS2 chunk")?
                    );
                }
            }
            SpeakBackend::Onnx => {
                let chunks = speech::synthesis_chunks_from_plan(&plan)?;
                for (index, chunk) in chunks.iter().enumerate() {
                    println!(
                        "  {}: {} (pause_after: {}ms)",
                        index + 1,
                        chunk.sequence.symbols.join(" "),
                        chunk.pause_after_ms
                    );
                }
            }
        }

        if !phonemicized.warnings.is_empty() {
            println!("warnings:");
            for warning in &phonemicized.warnings {
                println!("  {}", format_warning(warning));
            }
        }
        println!("sample_rate_hz: {}", artifact.sample_rate_hz);
        println!("samples: {}", artifact.pcm.len());
        if command.timings && !artifact.timings.is_empty() {
            println!("timings_ms:");
            for timing in &artifact.timings {
                println!("  {}: {:.2}", timing.stage, timing.elapsed_ms);
            }
        }

        *total_samples += artifact.pcm.len();
        all_pcm.extend(artifact.pcm);
        all_timings.extend(artifact.timings);
        Ok(())
    };

    if let Some(ref text) = command.text {
        process_chunk(
            text,
            &mut backend,
            &player,
            &mut all_pcm,
            &mut all_timings,
            &mut total_samples,
        )?;
    } else {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        let mut handle = stdin.lock();
        let mut line = String::new();
        while handle.read_line(&mut line)? > 0 {
            let sentences = split_into_sentences(&line);
            for sentence in sentences {
                if !sentence.is_empty() {
                    process_chunk(
                        &sentence,
                        &mut backend,
                        &player,
                        &mut all_pcm,
                        &mut all_timings,
                        &mut total_samples,
                    )?;
                }
            }
            line.clear();
        }
    }

    if let Some(ref output_path) = command.output {
        write_wav_mono_f32(output_path, target_sample_rate, &all_pcm)
            .with_context(|| format!("failed to write WAV to {}", output_path.display()))?;
        println!("wav: {}", output_path.display());
    } else {
        println!("Playing audio out loud via CPAL...");
        let playback_started = Instant::now();
        if let Some(ref p) = player {
            p.wait_until_done(total_samples);
        }
        if command.timings {
            println!(
                "playback_profile_json: {}",
                serde_json::json!({
                    "wait_ms": playback_started.elapsed().as_secs_f64() * 1_000.0,
                    "samples": total_samples,
                    "reported_separately_from_synthesis": true,
                })
            );
        }
        println!("wav: <none> (played out loud)");
    }

    Ok(())
}

fn load_burn_pipeline<B: Backend>(
    acoustic_config: &Path,
    acoustic_checkpoint: &Path,
    vocoder_config: &Path,
    vocoder_checkpoint: &Path,
    device: B::Device,
    profiler: &mut dyn FnMut(speech::ModelLoadProfileEvent),
) -> Result<speech::BurnSpeedySpeechPipeline<B>>
where
    B::Device: Clone,
{
    let acoustic = speech::BurnSpeedySpeechAcoustic::load_profiled(
        acoustic_config,
        acoustic_checkpoint,
        device.clone(),
        &mut *profiler,
    )
    .context("failed to load Burn SpeedySpeech acoustic model")?;
    let vocoder = speech::BurnHifiganVocoder::load_profiled(
        vocoder_config,
        vocoder_checkpoint,
        device,
        &mut *profiler,
    )
    .context("failed to load Burn HiFi-GAN vocoder")?;
    speech::BurnSpeedySpeechPipeline::new(acoustic, vocoder)
        .context("Burn speech components are incompatible")
}

fn print_available_speakers(command: &SpeakCommand) -> Result<()> {
    if matches!(command.backend, SpeakBackend::Burn) {
        println!("speakers: <none> (single-speaker acoustic model)");
        return Ok(());
    }
    if matches!(command.backend, SpeakBackend::Vits) {
        let checkpoint = crate::models::ensure_model_available(
            crate::models::DEFAULT_END_TO_END_SPEECH_MODEL_ID,
        )?;
        let config = speech::VitsInferenceConfig::from_file(component_config_path(&checkpoint)?)?;
        let speaker_path = checkpoint
            .parent()
            .context("VITS checkpoint path has no parent directory")?
            .join("speaker_ids.json");
        let catalog = speech::SpeakerCatalog::from_file(speaker_path, config.network.num_speakers)?;
        println!("speakers:");
        for speaker in catalog.available_names() {
            println!("  {speaker}");
        }
        return Ok(());
    }
    anyhow::ensure!(
        matches!(command.backend, SpeakBackend::Onnx),
        "--list-speakers is currently available for --backend onnx"
    );
    #[cfg(feature = "onnx-tts")]
    {
        use speech::{voice_config_path, VoiceConfig};
        let primary_model = crate::models::ensure_voice_model_available()?;
        let config_path = voice_config_path(&primary_model);
        let config = VoiceConfig::from_json_file(&config_path)?;
        let speakers = config.available_speaker_names();
        if speakers.is_empty() {
            println!("speakers: <none>");
        } else {
            println!("speakers:");
            for speaker in speakers {
                println!("  {speaker}");
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "onnx-tts"))]
    {
        anyhow::bail!("ONNX speech backend requires compiling with feature `onnx-tts`")
    }
}

fn component_config_path(checkpoint_path: &Path) -> Result<PathBuf> {
    let config_path = checkpoint_path
        .parent()
        .context("speech component checkpoint has no parent directory")?
        .join("config.json");
    anyhow::ensure!(
        config_path.is_file(),
        "speech component config is missing: {}",
        config_path.display()
    );
    Ok(config_path)
}

pub(crate) fn utterance_plan_from_phonemicized(output: &PhonemicizeOutput) -> UtterancePlan {
    UtterancePlan {
        id: UtteranceId("tongues.speak.utterance".into()),
        variety: output.variety.clone(),
        speaker: None,
        intended_text: Some(output.text.clone()),
        intended_morphemes: Vec::new(),
        intended_phonemes: output.phonemes.clone(),
        target_phones: output.phones.clone(),
        target_syllables: output.syllables.clone(),
        boundaries: output.boundaries.clone(),
        target_prosody: output.prosody.clone(),
        target_acoustics: Vec::new(),
        speaker_reference: None,
        style: None,
        provenance: EvidenceProvenance {
            source: EvidenceSource::TtsPlan,
            method: "tongues speak phonemicized plan".into(),
            version: Some("0.1".into()),
        },
    }
}

fn styletts2_options_from(max_tts_symbols: usize, no_tts_chunking: bool) -> StyleTts2PlanOptions {
    StyleTts2PlanOptions {
        max_symbols_per_chunk: max_tts_symbols,
        chunking_enabled: !no_tts_chunking,
    }
}

fn is_guessed_pronunciation(warning: &PronunciationWarning) -> bool {
    matches!(
        warning.kind,
        PronunciationWarningKind::GuessedWord
            | PronunciationWarningKind::MixedAlphaNumeric
            | PronunciationWarningKind::UnknownPronunciation
    )
}

fn format_warning(warning: &PronunciationWarning) -> String {
    if is_guessed_pronunciation(warning) {
        format!("guessed pronunciation: {}", warning.token)
    } else {
        warning.message.clone()
    }
}

fn format_phonemes(output: &PhonemicizeOutput) -> String {
    let symbols = output
        .phonemes
        .iter()
        .filter_map(|token| match &token.phoneme {
            Spec::Known(id) => Some((
                phoneme_default_phone_display_symbol(id, &output.variety),
                token_word_index(&token.features),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    format_symbols_with_boundary_markers(symbols, &output.boundaries)
}

fn format_phones(output: &PhonemicizeOutput) -> String {
    let symbols = output
        .phones
        .iter()
        .filter_map(|token| match &token.phone {
            Spec::Known(id) if !id.as_str().starts_with("boundary.") => Some((
                phone_display_symbol(id).to_string(),
                token_word_index(&token.features),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    format_symbols_with_boundary_markers(symbols, &output.boundaries)
}

fn format_symbols_with_boundary_markers(
    symbols: Vec<(String, Option<usize>)>,
    boundaries: &[SpeechBoundaryToken],
) -> String {
    let mut formatted = Vec::with_capacity(symbols.len());
    for (index, (mut symbol, word_index)) in symbols.iter().cloned().enumerate() {
        let next_word_index = symbols
            .get(index + 1)
            .and_then(|(_, word_index)| *word_index);
        if let Some(word_index) =
            word_index.filter(|word_index| Some(*word_index) != next_word_index)
        {
            for marker in boundary_markers_after_word(boundaries, word_index) {
                symbol.push_str(marker);
            }
        }
        formatted.push(symbol);
    }
    formatted.join(" ")
}

fn boundary_markers_after_word(
    boundaries: &[SpeechBoundaryToken],
    word_index: usize,
) -> impl Iterator<Item = &'static str> + '_ {
    boundaries
        .iter()
        .filter(move |boundary| boundary.after_grapheme_index == word_index)
        .filter_map(boundary_intonation_marker)
}

fn boundary_intonation_marker(boundary: &SpeechBoundaryToken) -> Option<&'static str> {
    if let Some(terminal) = boundary.terminal {
        return Some(match terminal {
            TerminalPunctuation::Question => "↗",
            TerminalPunctuation::Period | TerminalPunctuation::Exclamation => "↘",
        });
    }
    if let Some(pause) = boundary.pause {
        return Some(match pause {
            PauseKind::Comma => "→",
            PauseKind::AlternativeQuestionRise => "↗",
        });
    }
    None
}

fn token_word_index(features: &speaking::FeatureBundle) -> Option<usize> {
    let value = features
        .values
        .get(&FeatureId("orthography.word_index".into()))?;
    match value {
        Spec::Known(FeatureValue::Number(value)) if value.is_finite() && *value >= 0.0 => {
            Some(*value as usize)
        }
        _ => None,
    }
}

fn format_phonemes_with_features(output: &PhonemicizeOutput) -> String {
    output
        .phonemes
        .iter()
        .filter_map(|token| match &token.phoneme {
            Spec::Known(id) => {
                let symbol = phoneme_default_phone_display_symbol(id, &output.variety);
                let stress = token_feature_category(token, "stress");
                let reduced = token_feature_bool(token, "reduced_vowel");
                let mut annotations = Vec::new();
                if let Some(stress) = stress {
                    annotations.push(stress.to_string());
                }
                if reduced == Some(true) {
                    annotations.push("reduced".into());
                }
                if annotations.is_empty() {
                    Some(symbol)
                } else {
                    Some(format!("{symbol}({})", annotations.join(",")))
                }
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn token_feature_category<'a>(token: &'a speaking::PhonemeToken, name: &str) -> Option<&'a str> {
    let value = token
        .features
        .values
        .get(&speaking::FeatureId(format!("phonology.{name}")))?;
    match value {
        Spec::Known(speaking::FeatureValue::Category(value)) => Some(value.as_str()),
        Spec::Known(speaking::FeatureValue::Text(value)) => Some(value.as_str()),
        _ => None,
    }
}

fn token_feature_bool(token: &speaking::PhonemeToken, name: &str) -> Option<bool> {
    let value = token
        .features
        .values
        .get(&speaking::FeatureId(format!("phonology.{name}")))?;
    match value {
        Spec::Known(speaking::FeatureValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

pub(crate) fn write_wav_mono_f32(path: &Path, sample_rate_hz: u32, samples: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_sample(pcm)?;
    }
    writer.finalize()?;
    Ok(())
}

pub(crate) struct AudioStreamPlayer {
    samples: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    cursor: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    device_name: String,
    output_sample_rate_hz: u32,
    channels: u16,
    sample_format: cpal::SampleFormat,
    _stream: cpal::Stream,
}

impl AudioStreamPlayer {
    pub(crate) fn new(input_sample_rate: u32) -> Result<Self> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        };

        let host = cpal::default_host();
        let device = match host.default_output_device() {
            Some(d) => d,
            None => {
                anyhow::bail!("No default audio output device available.");
            }
        };
        let device_name = device.name().unwrap_or_else(|_| "<unknown>".to_string());

        let config = match device.default_output_config() {
            Ok(c) => c,
            Err(e) => {
                anyhow::bail!(
                    "Failed to get default output config for {}: {}",
                    device_name,
                    e
                );
            }
        };
        let sample_format = config.sample_format();
        let output_sample_rate = config.sample_rate().0;
        let channels = config.channels();

        let samples = Arc::new(Mutex::new(Vec::new()));
        let cursor = Arc::new(AtomicUsize::new(0));

        let cursor_clone = Arc::clone(&cursor);
        let samples_clone = Arc::clone(&samples);
        let resample_ratio = input_sample_rate as f64 / output_sample_rate as f64;

        let err_fn = |err| eprintln!("output stream error: {err}");
        let stream_config = config.config();

        let mut input_cursor: f64 = 0.0;

        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &stream_config,
                move |output: &mut [f32], _| {
                    let guard = samples_clone.lock().unwrap();
                    let mut frame_idx = 0;
                    while frame_idx < output.len() {
                        let left = input_cursor.floor() as usize;
                        if !guard.is_empty() && left < guard.len() {
                            let right = (left + 1).min(guard.len() - 1);
                            let fraction = (input_cursor - left as f64) as f32;
                            for chan in 0..channels {
                                let sample: f32 =
                                    guard[left] * (1.0_f32 - fraction) + guard[right] * fraction;
                                if let Some(out) = output.get_mut(frame_idx + chan as usize) {
                                    *out = sample;
                                }
                            }
                            input_cursor += resample_ratio;
                        } else {
                            for chan in 0..channels {
                                if let Some(out) = output.get_mut(frame_idx + chan as usize) {
                                    *out = 0.0;
                                }
                            }
                        }
                        frame_idx += channels as usize;
                    }
                    cursor_clone.store(input_cursor as usize, Ordering::Relaxed);
                },
                err_fn,
                None,
            )?,
            cpal::SampleFormat::I16 => device.build_output_stream(
                &stream_config,
                move |output: &mut [i16], _| {
                    let guard = samples_clone.lock().unwrap();
                    let mut frame_idx = 0;
                    while frame_idx < output.len() {
                        let left = input_cursor.floor() as usize;
                        if !guard.is_empty() && left < guard.len() {
                            let right = (left + 1).min(guard.len() - 1);
                            let fraction = (input_cursor - left as f64) as f32;
                            for chan in 0..channels {
                                let sample: f32 =
                                    guard[left] * (1.0_f32 - fraction) + guard[right] * fraction;
                                let sample_i16 = (sample * i16::MAX as f32)
                                    .clamp(i16::MIN as f32, i16::MAX as f32)
                                    as i16;
                                if let Some(out) = output.get_mut(frame_idx + chan as usize) {
                                    *out = sample_i16;
                                }
                            }
                            input_cursor += resample_ratio;
                        } else {
                            for chan in 0..channels {
                                if let Some(out) = output.get_mut(frame_idx + chan as usize) {
                                    *out = 0;
                                }
                            }
                        }
                        frame_idx += channels as usize;
                    }
                    cursor_clone.store(input_cursor as usize, Ordering::Relaxed);
                },
                err_fn,
                None,
            )?,
            cpal::SampleFormat::U16 => device.build_output_stream(
                &stream_config,
                move |output: &mut [u16], _| {
                    let guard = samples_clone.lock().unwrap();
                    let mut frame_idx = 0;
                    while frame_idx < output.len() {
                        let left = input_cursor.floor() as usize;
                        if !guard.is_empty() && left < guard.len() {
                            let right = (left + 1).min(guard.len() - 1);
                            let fraction = (input_cursor - left as f64) as f32;
                            for chan in 0..channels {
                                let sample: f32 =
                                    guard[left] * (1.0_f32 - fraction) + guard[right] * fraction;
                                let val = ((sample + 1.0_f32) * 0.5_f32 * u16::MAX as f32)
                                    .clamp(0.0_f32, u16::MAX as f32)
                                    as u16;
                                if let Some(out) = output.get_mut(frame_idx + chan as usize) {
                                    *out = val;
                                }
                            }
                            input_cursor += resample_ratio;
                        } else {
                            for chan in 0..channels {
                                if let Some(out) = output.get_mut(frame_idx + chan as usize) {
                                    *out = 32768;
                                }
                            }
                        }
                        frame_idx += channels as usize;
                    }
                    cursor_clone.store(input_cursor as usize, Ordering::Relaxed);
                },
                err_fn,
                None,
            )?,
            _ => anyhow::bail!("Unsupported CPAL sample format: {:?}", sample_format),
        };

        stream.play().context("failed to play CPAL stream")?;

        Ok(Self {
            samples,
            cursor,
            device_name,
            output_sample_rate_hz: output_sample_rate,
            channels,
            sample_format,
            _stream: stream,
        })
    }

    pub(crate) fn description(&self) -> String {
        format!(
            "{} format={:?} rate={}Hz channels={}",
            self.device_name, self.sample_format, self.output_sample_rate_hz, self.channels
        )
    }

    pub(crate) fn append(&self, chunk: &[f32]) {
        let mut guard = self.samples.lock().unwrap();
        guard.extend_from_slice(chunk);
    }

    pub(crate) fn wait_until_done(&self, input_sample_count: usize) {
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        while self.cursor.load(Ordering::Relaxed) < input_sample_count {
            std::thread::sleep(Duration::from_millis(50));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}
