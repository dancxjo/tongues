
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use clap::Args;

use speech::{
    EnglishPhonemicizer, EvidenceProvenance, EvidenceSource, FeatureId, FeatureValue, PauseKind,
    PhonemicizeOutput, PhonemicizeRequest, Phonemicizer, PronunciationWarning,
    PronunciationWarningKind, ProsodyTrack, Spec, SpeechBoundaryToken, TerminalPunctuation,
    UtteranceId, UtterancePlan, VarietyId, phone_display_symbol,
    phoneme_default_phone_display_symbol,
};
use styletts2::{
    BackendSynthesisPlan, DEFAULT_MAX_TTS_SYMBOLS, MockStyleTts2Backend, StyleTts2Backend,
    StyleTts2PlanOptions, StyleTts2SynthesisRequest, StyleTts2Timing, prepare_styletts2_plan,
    styletts2_en_us_symbol_set, styletts2_text_for_symbols, validate_styletts2_plan,
};

#[cfg(feature = "styletts2-onnx")]
use styletts2::{StyleTts2DiffusionOptions, StyleTts2OnnxBackend};


const DEFAULT_STYLE_ALPHA: f32 = 0.3;
const DEFAULT_STYLE_BETA: f32 = 0.1;
const DEFAULT_SPEED: f64 = 1.8;

#[derive(Debug, Args, Clone)]
pub struct SpeakCommand {
    #[arg(help = "The text to speak. If not provided, reads from stdin.")]
    pub text: Option<String>,
    #[arg(long, default_value = "en-US")]
    pub variety: String,
    #[arg(long, value_enum, default_value_t = SpeakBackend::Styletts2)]
    pub backend: SpeakBackend,
    #[arg(long, short)]
    pub output: Option<PathBuf>,
    #[arg(long, default_value_t = 24_000)]
    pub sample_rate_hz: u32,
    #[arg(long)]
    pub voice_wav: Option<PathBuf>,
    #[arg(long)]
    pub style_wav: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = SpeakQuality::Balanced)]
    pub quality: SpeakQuality,
    #[arg(long)]
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
        default_value_t = 1.0,
        help = "StyleTTS2 diffusion embedding scale"
    )]
    pub embedding_scale: f64,
    #[arg(long, default_value_t = 0)]
    pub style_seed: u64,
    #[arg(long, default_value_t = DEFAULT_SPEED, help = "StyleTTS2 decoder speed multiplier")]
    pub speed: f64,
    #[arg(long)]
    pub debug_pronunciation: bool,
    #[arg(long)]
    pub timings: bool,
    #[arg(long, default_value_t = DEFAULT_MAX_TTS_SYMBOLS)]
    pub max_tts_symbols: usize,
    #[arg(long)]
    pub no_tts_chunking: bool,
    #[arg(long)]
    pub fail_on_guessed_pronunciation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SpeakBackend {
    Mock,
    Styletts2,
    Piper,
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

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSynthesisArtifact {
    pub sample_rate_hz: u32,
    pub pcm: Vec<f32>,
    pub timings: Vec<StyleTts2Timing>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSynthesisOptions {
    pub sample_rate_hz: u32,
    pub voice_wav: Option<PathBuf>,
    pub style_wav: Option<PathBuf>,
    pub diffusion_steps: usize,
    pub style_alpha: f32,
    pub style_beta: f32,
    pub embedding_scale: f64,
    pub style_seed: u64,
    pub speed: f64,
    pub max_tts_symbols: usize,
    pub no_tts_chunking: bool,
}

impl From<&SpeakCommand> for SpeechSynthesisOptions {
    fn from(command: &SpeakCommand) -> Self {
        Self {
            sample_rate_hz: command.sample_rate_hz,
            voice_wav: command.voice_wav.clone(),
            style_wav: command.style_wav.clone(),
            diffusion_steps: command.resolved_diffusion_steps(),
            style_alpha: command.resolved_style_alpha(),
            style_beta: command.resolved_style_beta(),
            embedding_scale: command.embedding_scale,
            style_seed: command.style_seed,
            speed: command.speed,
            max_tts_symbols: command.max_tts_symbols,
            no_tts_chunking: command.no_tts_chunking,
        }
    }
}

pub fn run_speak(command: SpeakCommand) -> Result<()> {
    let phonemicized = EnglishPhonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: command.text.clone(),
            variety: VarietyId(command.variety.clone()),
            style: None,
        })
        .context("failed to phonemicize text into a speech plan")?;
    
    let plan = utterance_plan_from_phonemicized(&phonemicized);
    
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

    let backend_label = match command.backend {
        SpeakBackend::Mock => "mock",
        SpeakBackend::Styletts2 => "styletts2",
        SpeakBackend::Piper => "piper",
    };

    let target_sample_rate = match command.backend {
        SpeakBackend::Mock => command.sample_rate_hz,
        SpeakBackend::Styletts2 => command.sample_rate_hz,
        SpeakBackend::Piper => 22050,
    };

    let player = if command.output.is_none() {
        match AudioStreamPlayer::new(target_sample_rate) {
            Ok(p) => Some(p),
            Err(e) => {
                println!("Warning: Failed to initialize audio player: {}. Playing audio will be skipped.", e);
                None
            }
        }
    } else {
        None
    };

    let mut audio_callback = |chunk: &[f32]| {
        if let Some(ref p) = player {
            p.append(chunk);
        }
    };
    let cb_arg: Option<&mut dyn FnMut(&[f32])> = Some(&mut audio_callback);

    let options = SpeechSynthesisOptions::from(&command);

    let artifact = match command.backend {
        SpeakBackend::Mock => {
            let styletts2_plan = prepare_styletts2_plan(
                &plan,
                &styletts2_en_us_symbol_set(),
                styletts2_options_from(command.max_tts_symbols, command.no_tts_chunking),
            )
            .context("failed to prepare StyleTTS2 synthesis plan")?;
            synthesize_backend_plan_with_mock_to_wav(
                styletts2_plan,
                command.sample_rate_hz,
                cb_arg,
            )?
        }
        SpeakBackend::Styletts2 => {
            let styletts2_plan = prepare_styletts2_plan(
                &plan,
                &styletts2_en_us_symbol_set(),
                styletts2_options_from(command.max_tts_symbols, command.no_tts_chunking),
            )
            .context("failed to prepare StyleTTS2 synthesis plan")?;
            let primary_model = crate::models::ensure_styletts2_model_available()?;
            synthesize_backend_plan_with_styletts2_to_wav(
                styletts2_plan,
                &plan,
                &primary_model,
                &options,
                cb_arg,
            )?
        }
        SpeakBackend::Piper => {
            let primary_model = crate::models::ensure_piper_voice_model_available()?;
            synthesize_plan_with_piper(&plan, &primary_model, cb_arg)?
        }
    };

    let backend_symbols = match command.backend {
        SpeakBackend::Mock | SpeakBackend::Styletts2 => {
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
                    styletts2_text_for_symbols(&chunk.symbols).map(|text| text.trim().to_string())
                })
                .collect::<Result<Vec<_>, _>>()
                .context("failed to format StyleTTS2 backend symbols")?
                .join(" || ")
        }
        SpeakBackend::Piper => {
            let sequence = crate::piper::piper_sequence_from_plan(&plan)?;
            sequence.symbols.join(" ")
        }
    };

    println!("Pronlex speech synthesis plan");
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
    println!("backend_symbols: {backend_symbols}");

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
        SpeakBackend::Piper => {
            let chunks = crate::piper::piper_synthesis_chunks_from_plan(&plan)?;
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

    if let Some(ref output_path) = command.output {
        write_wav_mono_f32(output_path, artifact.sample_rate_hz, &artifact.pcm)
            .with_context(|| format!("failed to write WAV to {}", output_path.display()))?;
        println!("wav: {}", output_path.display());
    } else {
        println!("Playing audio out loud via CPAL...");
        if let Some(ref p) = player {
            p.wait_until_done(artifact.pcm.len());
        }
        println!("wav: <none> (played out loud)");
    }

    Ok(())
}

fn utterance_plan_from_phonemicized(output: &PhonemicizeOutput) -> UtterancePlan {
    UtterancePlan {
        id: UtteranceId("styletts2.demo.utterance".into()),
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
        style: None,
        provenance: EvidenceProvenance {
            source: EvidenceSource::TtsPlan,
            method: "pronlex speak phonemicized StyleTTS2 plan".into(),
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
        if word_index.is_some() && word_index != next_word_index {
            for marker in boundary_markers_after_word(boundaries, word_index.expect("checked")) {
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

fn token_word_index(features: &speech::FeatureBundle) -> Option<usize> {
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

fn token_feature_category<'a>(token: &'a speech::PhonemeToken, name: &str) -> Option<&'a str> {
    let value = token
        .features
        .values
        .get(&speech::FeatureId(format!("phonology.{name}")))?;
    match value {
        Spec::Known(speech::FeatureValue::Category(value)) => Some(value.as_str()),
        Spec::Known(speech::FeatureValue::Text(value)) => Some(value.as_str()),
        _ => None,
    }
}

fn token_feature_bool(token: &speech::PhonemeToken, name: &str) -> Option<bool> {
    let value = token
        .features
        .values
        .get(&speech::FeatureId(format!("phonology.{name}")))?;
    match value {
        Spec::Known(speech::FeatureValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

fn write_wav_mono_f32(path: &Path, sample_rate_hz: u32, samples: &[f32]) -> Result<()> {
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

fn synthesize_backend_plan_with_mock_to_wav(
    backend_plan: BackendSynthesisPlan,
    sample_rate_hz: u32,
    on_audio: Option<&mut dyn FnMut(&[f32])>,
) -> Result<SpeechSynthesisArtifact> {
    validate_styletts2_plan(&backend_plan).context("invalid StyleTTS2 synthesis plan")?;
    let request = StyleTts2SynthesisRequest::from_backend_plan(
        backend_plan,
        None,
        None,
        ProsodyTrack::default(),
    );
    let mut backend = MockStyleTts2Backend::new(sample_rate_hz);
    let mut pcm_mono_f32 = Vec::new();
    let mut on_audio = on_audio;
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
    })
}



pub struct AudioStreamPlayer {
    samples: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    cursor: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    _stream: cpal::Stream,
}

impl AudioStreamPlayer {
    pub fn new(input_sample_rate: u32) -> Result<Self> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

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
                anyhow::bail!("Failed to get default output config for {}: {}", device_name, e);
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
            cpal::SampleFormat::F32 => {
                device.build_output_stream(
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
                                    let sample: f32 = guard[left] * (1.0_f32 - fraction) + guard[right] * fraction;
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
                )?
            }
            cpal::SampleFormat::I16 => {
                device.build_output_stream(
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
                                    let sample: f32 = guard[left] * (1.0_f32 - fraction) + guard[right] * fraction;
                                    let sample_i16 = (sample * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
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
                )?
            }
            cpal::SampleFormat::U16 => {
                device.build_output_stream(
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
                                    let sample: f32 = guard[left] * (1.0_f32 - fraction) + guard[right] * fraction;
                                    let val = ((sample + 1.0_f32) * 0.5_f32 * u16::MAX as f32).clamp(0.0_f32, u16::MAX as f32) as u16;
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
                )?
            }
            _ => anyhow::bail!("Unsupported CPAL sample format: {:?}", sample_format),
        };
        
        stream.play().context("failed to play CPAL stream")?;
        
        Ok(Self {
            samples,
            cursor,
            _stream: stream,
        })
    }

    pub fn append(&self, chunk: &[f32]) {
        let mut guard = self.samples.lock().unwrap();
        guard.extend_from_slice(chunk);
    }

    pub fn wait_until_done(&self, input_sample_count: usize) {
        use std::time::Duration;
        use std::sync::atomic::Ordering;
        while self.cursor.load(Ordering::Relaxed) < input_sample_count {
            std::thread::sleep(Duration::from_millis(50));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}


#[cfg(feature = "styletts2-onnx")]
fn synthesize_backend_plan_with_styletts2_to_wav(
    backend_plan: BackendSynthesisPlan,
    plan: &UtterancePlan,
    primary_model_path: &Path,
    options: &SpeechSynthesisOptions,
    on_audio: Option<&mut dyn FnMut(&[f32])>,
) -> Result<SpeechSynthesisArtifact> {
    let model_dir = primary_model_path
        .parent()
        .context("StyleTTS2 primary model path has no parent directory")?;

    let diffusion_opts = StyleTts2DiffusionOptions {
        diffusion_steps: options.diffusion_steps,
        alpha: options.style_alpha,
        beta: options.style_beta,
        embedding_scale: options.embedding_scale,
        seed: options.style_seed,
    };

    let mut backend = StyleTts2OnnxBackend::from_model_dir(model_dir)
        .context("failed to load native StyleTTS2 ONNX backend")?
        .with_diffusion_options(diffusion_opts)
        .context("invalid StyleTTS2 diffusion options")?
        .with_speed(options.speed)
        .context("invalid StyleTTS2 speed")?;

    let default_refs = crate::models::ensure_styletts2_default_reference_audio_available()?;
    let voice_ref = options.voice_wav.as_ref().unwrap_or(&default_refs.voice);
    let style_ref = options.style_wav.as_ref().unwrap_or(&default_refs.style);

    let request = StyleTts2SynthesisRequest::from_backend_plan(
        backend_plan,
        plan.speaker.clone(),
        plan.style.clone(),
        plan.target_prosody.clone(),
    )
    .with_speaker_reference_audio_uri(voice_ref.display().to_string())
    .with_style_reference_audio_uri(style_ref.display().to_string());

    let mut pcm_mono_f32 = Vec::new();
    let mut on_audio = on_audio;
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
    })
}

#[cfg(not(feature = "styletts2-onnx"))]
fn synthesize_backend_plan_with_styletts2_to_wav(
    _backend_plan: BackendSynthesisPlan,
    _plan: &UtterancePlan,
    _primary_model_path: &Path,
    _options: &SpeechSynthesisOptions,
    _on_audio: Option<&mut dyn FnMut(&[f32])>,
) -> Result<SpeechSynthesisArtifact> {
    anyhow::bail!("StyleTTS2 native backend requires compiling with feature `styletts2-onnx`")
}

#[cfg(feature = "piper-onnx")]
fn synthesize_plan_with_piper(
    plan: &UtterancePlan,
    model_path: &Path,
    on_audio: Option<&mut dyn FnMut(&[f32])>,
) -> Result<SpeechSynthesisArtifact> {
    use crate::piper::{PiperOnnxBackend, PiperVoiceConfig, piper_voice_config_path, PiperAudioChunk};

    let config_path = piper_voice_config_path(model_path);
    let config = PiperVoiceConfig::from_json_file(&config_path)?;
    let sample_rate_hz = config.sample_rate_hz;

    let mut backend = PiperOnnxBackend::load(model_path, config)?;
    let mut pcm_mono_f32 = Vec::new();
    let mut on_audio = on_audio;

    backend.synthesize_plan_streaming(plan, &mut |chunk: PiperAudioChunk| {
        pcm_mono_f32.extend(&chunk.pcm_mono_f32);
        if let Some(ref mut cb) = on_audio {
            cb(&chunk.pcm_mono_f32);
        }
        Ok(())
    })?;

    Ok(SpeechSynthesisArtifact {
        sample_rate_hz,
        pcm: pcm_mono_f32,
        timings: Vec::new(),
    })
}

#[cfg(not(feature = "piper-onnx"))]
fn synthesize_plan_with_piper(
    _plan: &UtterancePlan,
    _model_path: &Path,
    _on_audio: Option<&mut dyn FnMut(&[f32])>,
) -> Result<SpeechSynthesisArtifact> {
    anyhow::bail!("Piper native backend requires compiling with feature `piper-onnx`")
}



