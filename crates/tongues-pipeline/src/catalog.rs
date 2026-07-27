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
    /// Whether the port carries an incrementally produced sequence of values.
    ///
    /// Cardinality describes graph connections, not the number of values a
    /// producer may emit during one run.
    #[serde(default)]
    pub streaming: bool,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splitter: Option<SplitterSpec>,
    #[serde(default)]
    pub replacement: ReplacementSpec,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitterSpec {
    pub strategy: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Readiness {
    Ready,
    Unavailable,
    Unverified,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplacementSpec {
    /// Backend-owned semantic role. Equal port shapes alone are never enough
    /// to make two nodes replacement candidates.
    #[serde(default)]
    pub family: String,
    /// Stable identity for the configuration contract, independent of labels.
    #[serde(default)]
    pub configuration_schema_id: String,
    #[serde(default)]
    pub configuration_schema_version: u32,
    /// Explicit source-port to destination-port mappings for declared
    /// cross-kind replacements.
    #[serde(default)]
    pub port_aliases: BTreeMap<String, String>,
    /// Explicit source-field to destination-field configuration mappings.
    #[serde(default)]
    pub configuration_aliases: BTreeMap<String, String>,
    /// Source ports whose connections may be deliberately removed by a lossy
    /// plan. Missing ports otherwise fail closed.
    #[serde(default)]
    pub disconnect_ports: BTreeSet<String>,
}

impl ReplacementSpec {
    pub fn for_node_kind(kind: &str) -> Self {
        Self {
            family: kind.into(),
            configuration_schema_id: format!("tongues.pipeline.{kind}.config"),
            configuration_schema_version: 1,
            ..Self::default()
        }
    }
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
    #[serde(default)]
    pub replacement: ReplacementSpec,
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
            schema_version: 2,
            revision: "tongues-pipeline-catalog-v3".into(),
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
        streaming: false,
    }
}

fn stream_port(
    id: &str,
    direction: PortDirection,
    value_type: ValueType,
    cardinality: Cardinality,
    must_consume: bool,
) -> PortSpec {
    let mut spec = port(id, direction, value_type, cardinality, must_consume);
    spec.streaming = true;
    spec
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
        splitter: None,
        replacement: ReplacementSpec::for_node_kind(kind),
    }
}

fn configured_source(
    kind: &str,
    label: &str,
    ports: Vec<PortSpec>,
    field: ConfiguredStringField<'_>,
) -> NodeKindSpec {
    let mut spec = node(kind, label, ports);
    let mut field_schema = serde_json::json!({
        "type": "string",
        "title": field.title,
        "description": field.description,
        "minLength": 1
    });
    if let Some(format) = field.format {
        field_schema["format"] = serde_json::Value::String(format.into());
    }
    spec.configuration_schema = serde_json::json!({
        "type": "object",
        "properties": {field.name: field_schema},
        "required": [field.name]
    });
    spec.default_config = serde_json::json!({field.name: field.default});
    spec
}

struct ConfiguredStringField<'a> {
    name: &'a str,
    title: &'a str,
    description: &'a str,
    format: Option<&'a str>,
    default: &'a str,
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
        configured_source(
            "audio_file",
            "Audio file",
            vec![
                port("out", Output, AudioBuffer, Many, true),
                port("artifact", Output, Artifact, Many, false),
                port("error", Output, Error, Many, false),
            ],
            ConfiguredStringField {
                name: "path",
                title: "Audio file path",
                description: "Path to the audio file the graph should read.",
                format: None,
                default: "",
            },
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
        configured_source(
            "text_source",
            "Inline text",
            vec![stream_port("out", Output, Text, Many, true)],
            ConfiguredStringField {
                name: "text",
                title: "Text",
                description: "Text emitted as one event on the text stream when the graph runs.",
                format: Some("multiline"),
                default: "Hello from Tongues.",
            },
        ),
        configured_source(
            "text_file",
            "Text file",
            vec![
                stream_port("out", Output, Text, Many, true),
                port("error", Output, Error, Many, false),
            ],
            ConfiguredStringField {
                name: "path",
                title: "Workspace text file",
                description: "Workspace-relative UTF-8 file emitted incrementally on the text stream.",
                format: Some("path"),
                default: "",
            },
        ),
        configured_source(
            "text_url",
            "Text URL",
            vec![
                stream_port("out", Output, Text, Many, true),
                port("error", Output, Error, Many, false),
            ],
            ConfiguredStringField {
                name: "url",
                title: "HTTP(S) text URL",
                description: "HTTP(S) URL whose UTF-8 response body is emitted incrementally on the text stream.",
                format: Some("uri"),
                default: "",
            },
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
    let mut transcript_splitter = node(
        "transcript_splitter",
        "Transcript splitter",
        vec![
            port("in", Input, TranscriptCommitted, One, false),
            port("out", Output, TranscriptCommitted, Many, true),
        ],
    );
    transcript_splitter.splitter = Some(SplitterSpec {
        strategy: "copy_each_value_to_every_outgoing_edge".into(),
    });
    kinds.push(transcript_splitter);
    kinds
}
