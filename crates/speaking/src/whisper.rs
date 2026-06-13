#[cfg(feature = "asr-whisper")]
mod imp {
    use std::sync::OnceLock;

    use whisper_cpp_plus::whisper_cpp_plus_sys as whisper_ffi;
    use whisper_cpp_plus::{FullParams, SamplingStrategy, WhisperState};

    use crate::asr::{
        AudioFrame, SpeechRecognizer, StreamingPartialKind, StreamingRecognition,
        StreamingRecognizerBackend, StreamingSpeechRecognizer,
    };
    use crate::transcript::{
        TranscriptCandidateEvent, TranscriptCandidateTracker, TranscriptChunk,
    };
    use crate::word_stream::TranscriptWord;

    #[derive(Debug, Clone, PartialEq)]
    pub struct WhisperTranscript {
        pub text: String,
        pub words: Vec<TranscriptWord>,
    }

    pub struct WhisperSpeechRecognizer {
        ctx: whisper_cpp_plus::WhisperContext,
        pending: Vec<f32>,
        sample_rate_hz: u32,
        input_silence_padding_ms: u64,
        candidate_tracker: TranscriptCandidateTracker,
    }

    const DEFAULT_INPUT_SILENCE_PADDING_MS: u64 = 250;
    const WHISPER_ROLLING_WINDOW_BACKEND: StreamingRecognizerBackend = StreamingRecognizerBackend {
        source: "whisper_rolling_window",
        partial_kind: StreamingPartialKind::Approximate,
    };

    impl WhisperSpeechRecognizer {
        pub fn new(model_path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
            Self::new_with_log_suppression(model_path, false)
        }

        pub fn new_quiet(model_path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
            Self::new_with_log_suppression(model_path, true)
        }

        pub fn new_quiet_without_input_padding(
            model_path: impl AsRef<std::path::Path>,
        ) -> anyhow::Result<Self> {
            Self::new_with_log_suppression_and_padding(model_path, true, 0)
        }

        fn new_with_log_suppression(
            model_path: impl AsRef<std::path::Path>,
            suppress_logs: bool,
        ) -> anyhow::Result<Self> {
            Self::new_with_log_suppression_and_padding(
                model_path,
                suppress_logs,
                DEFAULT_INPUT_SILENCE_PADDING_MS,
            )
        }

        fn new_with_log_suppression_and_padding(
            model_path: impl AsRef<std::path::Path>,
            suppress_logs: bool,
            input_silence_padding_ms: u64,
        ) -> anyhow::Result<Self> {
            configure_whisper_logging(suppress_logs);
            let ctx = whisper_cpp_plus::WhisperContext::new(model_path.as_ref())?;
            Ok(Self {
                ctx,
                pending: Vec::new(),
                sample_rate_hz: 16_000,
                input_silence_padding_ms,
                candidate_tracker: TranscriptCandidateTracker::new(),
            })
        }

        fn accept_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<()> {
            anyhow::ensure!(
                frame.sample_rate_hz == self.sample_rate_hz,
                "Whisper expects {} Hz audio; got {} Hz",
                self.sample_rate_hz,
                frame.sample_rate_hz
            );
            anyhow::ensure!(
                frame.channels == 1,
                "Whisper expects mono audio; got {} channels",
                frame.channels
            );
            self.pending.extend_from_slice(&frame.samples);
            Ok(())
        }

        fn poll_transcript(&mut self) -> anyhow::Result<Option<WhisperTranscript>> {
            if self.pending.is_empty() {
                return Ok(None);
            }
            let audio = std::mem::take(&mut self.pending);
            let padding_ms = self.input_silence_padding_ms;
            let audio = pad_samples_with_silence(audio, self.sample_rate_hz, padding_ms);
            let mut state = WhisperState::new(&self.ctx)?;
            let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
                .token_timestamps(true)
                .split_on_word(true);
            state.full(params, &audio)?;
            let transcript = transcript_from_whisper_state(&state, padding_ms)?;
            let text = transcript.text.trim();
            if text.is_empty() {
                return Ok(None);
            }
            Ok(Some(WhisperTranscript {
                text: text.to_owned(),
                words: transcript.words,
            }))
        }

        pub fn poll_candidate_events(&mut self) -> anyhow::Result<Vec<TranscriptCandidateEvent>> {
            self.poll_candidate_events_with_finality(true)
        }

        pub fn poll_candidate_events_with_finality(
            &mut self,
            is_final: bool,
        ) -> anyhow::Result<Vec<TranscriptCandidateEvent>> {
            Ok(self.poll_streaming(is_final)?.candidate_events)
        }

        pub fn poll_timed_transcript_with_finality(
            &mut self,
            is_final: bool,
        ) -> anyhow::Result<StreamingRecognition> {
            self.poll_streaming(is_final)
        }
    }

    fn pad_samples_with_silence(audio: Vec<f32>, sample_rate_hz: u32, padding_ms: u64) -> Vec<f32> {
        if audio.is_empty() || sample_rate_hz == 0 || padding_ms == 0 {
            return audio;
        }
        let padding_samples = (u64::from(sample_rate_hz) * padding_ms).div_ceil(1_000) as usize;
        if padding_samples == 0 {
            return audio;
        }
        let mut padded = Vec::with_capacity(
            audio
                .len()
                .saturating_add(padding_samples.saturating_mul(2)),
        );
        padded.extend(std::iter::repeat_n(0.0, padding_samples));
        padded.extend(audio);
        padded.extend(std::iter::repeat_n(0.0, padding_samples));
        padded
    }

    fn configure_whisper_logging(suppress_logs: bool) {
        static LOGGING_CONFIGURED: OnceLock<()> = OnceLock::new();
        if suppress_logs || !developer_diagnostics_enabled() {
            LOGGING_CONFIGURED.get_or_init(|| unsafe {
                whisper_ffi::whisper_log_set(Some(drop_whisper_log), std::ptr::null_mut());
            });
        }
    }

    fn transcript_from_whisper_state(
        state: &WhisperState,
        padding_ms: u64,
    ) -> anyhow::Result<WhisperTranscript> {
        let n_segments = state.full_n_segments();
        let mut text = String::new();
        let mut words = Vec::new();
        for segment_index in 0..n_segments {
            let segment_text = state.full_get_segment_text(segment_index)?;
            if !segment_text.trim().is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(segment_text.trim());
            }
            for token_index in 0..state.full_n_tokens(segment_index) {
                let token_text = state.full_get_token_text(segment_index, token_index)?;
                let token_text = token_text.trim();
                if token_text.is_empty() || token_text.starts_with('[') {
                    continue;
                }
                let Some(token) = state.full_get_token_data(segment_index, token_index) else {
                    continue;
                };
                let start_ms = whisper_time_to_ms(token.t0).saturating_sub(padding_ms);
                let end_ms = whisper_time_to_ms(token.t1).saturating_sub(padding_ms);
                words.push(TranscriptWord {
                    text: token_text.to_string(),
                    start_ms: Some(start_ms),
                    end_ms: Some(end_ms.max(start_ms.saturating_add(1))),
                    confidence: token.p.is_finite().then(|| token.p.clamp(0.0, 1.0)),
                });
            }
        }
        Ok(WhisperTranscript { text, words })
    }

    fn whisper_time_to_ms(value: i64) -> u64 {
        u64::try_from(value).unwrap_or_default().saturating_mul(10)
    }

    fn developer_diagnostics_enabled() -> bool {
        std::env::var_os("SPEAKING_DEVELOPER_DIAGNOSTICS").is_some()
    }

    unsafe extern "C" fn drop_whisper_log(
        _level: whisper_ffi::ggml_log_level,
        _text: *const std::ffi::c_char,
        _user_data: *mut std::ffi::c_void,
    ) {
    }

    impl SpeechRecognizer for WhisperSpeechRecognizer {
        fn push_frame(&mut self, frame: &AudioFrame) -> anyhow::Result<()> {
            self.accept_frame(frame)
        }

        fn poll_chunks(&mut self) -> anyhow::Result<Vec<TranscriptChunk>> {
            let output = self.flush()?;
            if output.text.is_empty() {
                return Ok(Vec::new());
            }
            Ok(vec![TranscriptChunk {
                text: output.text,
                is_final: true,
            }])
        }
    }

    impl StreamingSpeechRecognizer for WhisperSpeechRecognizer {
        fn poll_streaming(&mut self, is_final: bool) -> anyhow::Result<StreamingRecognition> {
            let transcript = self.poll_transcript()?;
            let (text, words) = transcript
                .as_ref()
                .map(|transcript| (transcript.text.clone(), transcript.words.clone()))
                .unwrap_or_default();
            let candidate_events = if transcript.is_some() {
                self.candidate_tracker
                    .ingest_candidate(text.clone(), None, is_final)
            } else if is_final {
                self.candidate_tracker.cancel_active()
            } else {
                Vec::new()
            };
            Ok(StreamingRecognition {
                text,
                words,
                candidate_events,
                backend: self.backend(),
            })
        }

        fn backend(&self) -> StreamingRecognizerBackend {
            WHISPER_ROLLING_WINDOW_BACKEND
        }
    }

    #[cfg(test)]
    mod tests {
        use super::pad_samples_with_silence;

        #[test]
        fn pads_samples_with_silence_on_both_ends() {
            let padded = pad_samples_with_silence(vec![0.5, -0.5], 1_000, 2);
            assert_eq!(padded, vec![0.0, 0.0, 0.5, -0.5, 0.0, 0.0]);
        }
    }
}

#[cfg(feature = "asr-whisper")]
pub use imp::*;
