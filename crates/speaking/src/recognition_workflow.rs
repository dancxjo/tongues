//! Library-owned definitions for user-facing recognition workflows.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FriendlySpeechVerb {
    Listen,
    Transcribe,
    Recognize,
    Interpret,
    Converse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecognitionWorkflowStage {
    AudioInput,
    VoiceActivityDetection,
    Segmentation,
    AutomaticSpeechRecognition,
    TranscriptNormalization,
    SentenceBoundary,
    Interpretation,
    ResponseProvider,
    ResponseFormatting,
    TextToSpeech,
    AudioOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecognitionWorkflowDefinition {
    pub verb: FriendlySpeechVerb,
    pub result: String,
    pub stages: Vec<RecognitionWorkflowStage>,
}

pub fn recognition_workflow(verb: FriendlySpeechVerb) -> RecognitionWorkflowDefinition {
    use RecognitionWorkflowStage as Stage;
    let (result, stages) = match verb {
        FriendlySpeechVerb::Listen => ("audio_events", vec![Stage::AudioInput]),
        FriendlySpeechVerb::Transcribe => (
            "committed_text",
            vec![
                Stage::AudioInput,
                Stage::AutomaticSpeechRecognition,
                Stage::TranscriptNormalization,
            ],
        ),
        FriendlySpeechVerb::Recognize => (
            "structured_linguistic_output",
            vec![
                Stage::AudioInput,
                Stage::AutomaticSpeechRecognition,
                Stage::TranscriptNormalization,
                Stage::SentenceBoundary,
            ],
        ),
        FriendlySpeechVerb::Interpret => (
            "semantic_output",
            vec![
                Stage::AudioInput,
                Stage::AutomaticSpeechRecognition,
                Stage::TranscriptNormalization,
                Stage::SentenceBoundary,
                Stage::Interpretation,
            ],
        ),
        FriendlySpeechVerb::Converse => (
            "response_audio",
            vec![
                Stage::AudioInput,
                Stage::AutomaticSpeechRecognition,
                Stage::TranscriptNormalization,
                Stage::SentenceBoundary,
                Stage::Interpretation,
                Stage::ResponseProvider,
                Stage::ResponseFormatting,
                Stage::TextToSpeech,
                Stage::AudioOutput,
            ],
        ),
    };
    RecognitionWorkflowDefinition {
        verb,
        result: result.into(),
        stages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn friendly_verbs_have_library_owned_semantics() {
        assert_eq!(
            recognition_workflow(FriendlySpeechVerb::Transcribe).result,
            "committed_text"
        );
        assert!(
            recognition_workflow(FriendlySpeechVerb::Recognize)
                .stages
                .contains(&RecognitionWorkflowStage::SentenceBoundary)
        );
        let converse = recognition_workflow(FriendlySpeechVerb::Converse);
        assert!(
            converse
                .stages
                .contains(&RecognitionWorkflowStage::ResponseProvider)
        );
        assert!(
            converse
                .stages
                .contains(&RecognitionWorkflowStage::TextToSpeech)
        );
        assert!(
            converse
                .stages
                .contains(&RecognitionWorkflowStage::AudioOutput)
        );
    }
}
