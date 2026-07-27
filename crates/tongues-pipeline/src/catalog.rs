use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueType {
    AudioStream,
    AudioBuffer,
    Text,
    TranscriptPartial,
    TranscriptRevised,
    TranscriptCommitted,
    Language,
    SpeakerAssignment,
    UtterancePlan,
    Control,
    Cancellation,
    Artifact,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortDirection {
    Input,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cardinality {
    One,
    Optional,
    Many,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortSpec {
    pub id: String,
    pub label: String,
    pub direction: PortDirection,
    pub value_type: ValueType,
    pub cardinality: Cardinality,
    #[serde(default)]
    pub must_consume: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeKindSpec {
    pub kind: String,
    pub label: String,
    #[serde(default)]
    pub ports: Vec<PortSpec>,
    #[serde(default)]
    pub configuration_schema: Value,
    #[serde(default)]
    pub default_config: Value,
    #[serde(default)]
    pub required_capabilities: BTreeSet<String>,
    #[serde(default)]
    pub requires_component: bool,
    #[serde(default)]
    pub unsafe_execution: bool,
    #[serde(default)]
    pub permits_cycle: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<AdapterSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge: Option<MergeSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterSpec {
    pub from: ValueType,
    pub to: ValueType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeSpec {
    pub strategy: String,
    pub deterministic_order: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Readiness {
    Ready,
    Unavailable,
    Unverified,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentSpec {
    pub id: String,
    pub node_kind: String,
    pub provider: String,
    pub model: String,
    pub readiness: Readiness,
    #[serde(default)]
    pub capabilities: BTreeSet<String>,
    #[serde(default)]
    pub configuration_schema: Value,
    #[serde(default)]
    pub default_config: Value,
    #[serde(default)]
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphCatalog {
    pub schema_version: u32,
    pub revision: String,
    pub node_kinds: BTreeMap<String, NodeKindSpec>,
    #[serde(default)]
    pub components: BTreeMap<String, ComponentSpec>,
}

impl GraphCatalog {
    pub fn builtin() -> Self {
        let mut catalog = Self {
            schema_version: 1,
            revision: "tongues-pipeline-catalog-v1".into(),
            node_kinds: BTreeMap::new(),
            components: BTreeMap::new(),
        };
        for spec in builtin_node_kinds() {
            catalog.node_kinds.insert(spec.kind.clone(), spec);
        }
        catalog
    }

    pub fn register_component(&mut self, component: ComponentSpec) {
        self.components.insert(component.id.clone(), component);
    }

    pub fn adapters_for(&self, from: ValueType, to: ValueType) -> Vec<String> {
        self.node_kinds
            .values()
            .filter_map(|kind| {
                kind.adapter
                    .as_ref()
                    .filter(|adapter| adapter.from == from && adapter.to == to)
                    .map(|_| kind.kind.clone())
            })
            .collect()
    }
}

fn port(
    id: &str,
    direction: PortDirection,
    value_type: ValueType,
    cardinality: Cardinality,
    must_consume: bool,
) -> PortSpec {
    PortSpec {
        id: id.into(),
        label: id.replace('_', " "),
        direction,
        value_type,
        cardinality,
        must_consume,
    }
}

fn node(kind: &str, label: &str, ports: Vec<PortSpec>) -> NodeKindSpec {
    NodeKindSpec {
        kind: kind.into(),
        label: label.into(),
        ports,
        configuration_schema: serde_json::json!({"type":"object"}),
        default_config: serde_json::json!({}),
        required_capabilities: BTreeSet::new(),
        requires_component: false,
        unsafe_execution: false,
        permits_cycle: false,
        adapter: None,
        merge: None,
    }
}

fn io(input: ValueType, output: ValueType) -> Vec<PortSpec> {
    vec![
        port("in", PortDirection::Input, input, Cardinality::One, false),
        port(
            "out",
            PortDirection::Output,
            output,
            Cardinality::Many,
            true,
        ),
        port(
            "error",
            PortDirection::Output,
            ValueType::Error,
            Cardinality::Many,
            false,
        ),
    ]
}

fn component_node(kind: &str, label: &str, ports: Vec<PortSpec>, capability: &str) -> NodeKindSpec {
    let mut spec = node(kind, label, ports);
    spec.requires_component = true;
    spec.required_capabilities.insert(capability.into());
    spec
}

fn builtin_node_kinds() -> Vec<NodeKindSpec> {
    use Cardinality::{Many, One, Optional};
    use PortDirection::{Input, Output};
    use ValueType::*;
    let mut kinds = vec![
        node(
            "microphone",
            "Microphone",
            vec![
                port("out", Output, AudioStream, Many, true),
                port("cancel", Input, Cancellation, Optional, false),
                port("error", Output, Error, Many, false),
            ],
        ),
        node(
            "audio_file",
            "Audio file",
            vec![
                port("out", Output, AudioBuffer, Many, true),
                port("artifact", Output, Artifact, Many, false),
                port("error", Output, Error, Many, false),
            ],
        ),
        node(
            "vad",
            "Voice activity detection",
            io(AudioStream, AudioStream),
        ),
        component_node(
            "asr",
            "Speech recognition",
            vec![
                port("audio", Input, AudioStream, One, false),
                port("partial", Output, TranscriptPartial, Many, false),
                port("revised", Output, TranscriptRevised, Many, false),
                port("committed", Output, TranscriptCommitted, Many, true),
                port("language", Output, Language, Many, false),
                port("error", Output, Error, Many, false),
            ],
            "asr",
        ),
        component_node(
            "diarization",
            "Speaker diarization",
            vec![
                port("audio", Input, AudioStream, One, false),
                port("speakers", Output, SpeakerAssignment, Many, true),
                port("error", Output, Error, Many, false),
            ],
            "diarization",
        ),
        node(
            "diarized_transcript",
            "Diarized transcript join",
            vec![
                port("transcript", Input, TranscriptCommitted, One, false),
                port("speakers", Input, SpeakerAssignment, One, false),
                port("out", Output, TranscriptCommitted, Many, true),
                port("error", Output, Error, Many, false),
            ],
        ),
        node(
            "text_source",
            "Text source",
            vec![port("out", Output, Text, Many, true)],
        ),
        node(
            "linguistic",
            "Linguistic and prosody analysis",
            io(Text, UtterancePlan),
        ),
        component_node(
            "response",
            "Response generation",
            io(Text, Text),
            "text_generation",
        ),
        component_node(
            "interpretation",
            "Spoken-language interpretation",
            io(Text, Text),
            "interpretation",
        ),
        component_node(
            "tts",
            "Speech synthesis",
            vec![
                port("in", Input, Text, Optional, false),
                port("plan", Input, UtterancePlan, Optional, false),
                port("out", Output, AudioStream, Many, true),
                port("error", Output, Error, Many, false),
            ],
            "tts",
        ),
        node(
            "audio_output",
            "Audio output",
            vec![
                port("in", Input, AudioStream, One, false),
                port("played", Output, Artifact, Many, false),
                port("cancel", Input, Cancellation, Optional, false),
                port("error", Output, Error, Many, false),
            ],
        ),
        node(
            "transcript_sink",
            "Transcript output",
            vec![
                port("in", Input, TranscriptCommitted, One, false),
                port("artifact", Output, Artifact, Many, false),
            ],
        ),
        node(
            "control_source",
            "Control source",
            vec![
                port("control", Output, Control, Many, false),
                port("cancel", Output, Cancellation, Many, false),
            ],
        ),
    ];
    let mut audio_adapter = node(
        "audio_buffer_to_stream",
        "Audio buffer to stream",
        io(AudioBuffer, AudioStream),
    );
    audio_adapter.adapter = Some(AdapterSpec {
        from: AudioBuffer,
        to: AudioStream,
    });
    kinds.push(audio_adapter);
    let mut text_adapter = node(
        "committed_transcript_to_text",
        "Committed transcript to text",
        io(TranscriptCommitted, Text),
    );
    text_adapter.adapter = Some(AdapterSpec {
        from: TranscriptCommitted,
        to: Text,
    });
    kinds.push(text_adapter);
    let mut transcript_merge = node(
        "transcript_merge",
        "Transcript merge",
        vec![
            port("in", Input, TranscriptCommitted, Many, false),
            port("out", Output, TranscriptCommitted, Many, true),
        ],
    );
    transcript_merge.merge = Some(MergeSpec {
        strategy: "event_time_then_edge_id".into(),
        deterministic_order: "event time, source edge ID, source sequence".into(),
    });
    kinds.push(transcript_merge);
    kinds
}
