use crate::{
    Endpoint, GraphCatalog, GraphDocument, GraphEdge, GraphNode, Readiness, default_edge_capacity,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StarterGraph {
    TextToSpeech,
    Transcription,
    MeetingTranscription,
    Interpretation,
    LiveConversation,
}

impl StarterGraph {
    pub const ALL: [Self; 5] = [
        Self::TextToSpeech,
        Self::Transcription,
        Self::MeetingTranscription,
        Self::Interpretation,
        Self::LiveConversation,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::TextToSpeech => "text_to_speech",
            Self::Transcription => "transcription",
            Self::MeetingTranscription => "meeting_transcription",
            Self::Interpretation => "interpretation",
            Self::LiveConversation => "live_conversation",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::TextToSpeech => "Text to speech",
            Self::Transcription => "Transcription",
            Self::MeetingTranscription => "Meeting transcription with diarization",
            Self::Interpretation => "Spoken interpretation",
            Self::LiveConversation => "Live conversation",
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StarterError {
    #[error("starter graph `{starter}` needs a ready `{node_kind}` component")]
    MissingComponent {
        starter: &'static str,
        node_kind: &'static str,
    },
}

pub fn starter_graph(
    starter: StarterGraph,
    catalog: &GraphCatalog,
) -> Result<GraphDocument, StarterError> {
    let mut graph = GraphDocument::new(
        format!("starter:{}", starter.id()),
        starter.label().to_owned(),
    );
    match starter {
        StarterGraph::TextToSpeech => {
            add(&mut graph, "text", "text_source", None);
            add(
                &mut graph,
                "tts",
                "tts",
                Some(component(starter, catalog, "tts")?),
            );
            add(&mut graph, "audio", "audio_output", None);
            edge(&mut graph, "text", "out", "tts", "in");
            edge(&mut graph, "tts", "out", "audio", "in");
            select(&mut graph, "audio", "in");
        }
        StarterGraph::Transcription => {
            add(&mut graph, "microphone", "microphone", None);
            add(
                &mut graph,
                "asr",
                "asr",
                Some(component(starter, catalog, "asr")?),
            );
            add(&mut graph, "transcript", "transcript_sink", None);
            edge(&mut graph, "microphone", "out", "asr", "audio");
            edge(&mut graph, "asr", "committed", "transcript", "in");
            select(&mut graph, "transcript", "in");
        }
        StarterGraph::MeetingTranscription => {
            add(&mut graph, "microphone", "microphone", None);
            add(&mut graph, "vad", "vad", None);
            add(
                &mut graph,
                "asr",
                "asr",
                Some(component(starter, catalog, "asr")?),
            );
            add(
                &mut graph,
                "diarization",
                "diarization",
                Some(component(starter, catalog, "diarization")?),
            );
            add(&mut graph, "join", "diarized_transcript", None);
            add(&mut graph, "transcript", "transcript_sink", None);
            edge(&mut graph, "microphone", "out", "vad", "in");
            edge(&mut graph, "vad", "out", "asr", "audio");
            edge(&mut graph, "vad", "out", "diarization", "audio");
            edge(&mut graph, "asr", "committed", "join", "transcript");
            edge(&mut graph, "diarization", "speakers", "join", "speakers");
            edge(&mut graph, "join", "out", "transcript", "in");
            select(&mut graph, "transcript", "in");
        }
        StarterGraph::Interpretation => {
            add(&mut graph, "microphone", "microphone", None);
            add(
                &mut graph,
                "asr",
                "asr",
                Some(component(starter, catalog, "asr")?),
            );
            add(&mut graph, "to_text", "committed_transcript_to_text", None);
            add(
                &mut graph,
                "interpret",
                "interpretation",
                Some(component(starter, catalog, "interpretation")?),
            );
            add(
                &mut graph,
                "tts",
                "tts",
                Some(component(starter, catalog, "tts")?),
            );
            add(&mut graph, "audio", "audio_output", None);
            edge(&mut graph, "microphone", "out", "asr", "audio");
            edge(&mut graph, "asr", "committed", "to_text", "in");
            edge(&mut graph, "to_text", "out", "interpret", "in");
            edge(&mut graph, "interpret", "out", "tts", "in");
            edge(&mut graph, "tts", "out", "audio", "in");
            select(&mut graph, "audio", "in");
        }
        StarterGraph::LiveConversation => {
            add(&mut graph, "microphone", "microphone", None);
            add(
                &mut graph,
                "asr",
                "asr",
                Some(component(starter, catalog, "asr")?),
            );
            add(&mut graph, "to_text", "committed_transcript_to_text", None);
            add(
                &mut graph,
                "response",
                "response",
                Some(component(starter, catalog, "response")?),
            );
            add(
                &mut graph,
                "tts",
                "tts",
                Some(component(starter, catalog, "tts")?),
            );
            add(&mut graph, "audio", "audio_output", None);
            edge(&mut graph, "microphone", "out", "asr", "audio");
            edge(&mut graph, "asr", "committed", "to_text", "in");
            edge(&mut graph, "to_text", "out", "response", "in");
            edge(&mut graph, "response", "out", "tts", "in");
            edge(&mut graph, "tts", "out", "audio", "in");
            select(&mut graph, "audio", "in");
        }
    }
    for node in &mut graph.nodes {
        let defaults = node
            .component_id
            .as_ref()
            .and_then(|id| catalog.components.get(id))
            .map(|component| &component.default_config)
            .or_else(|| {
                catalog
                    .node_kinds
                    .get(&node.kind)
                    .map(|kind| &kind.default_config)
            });
        if let Some(serde_json::Value::Object(defaults)) = defaults {
            node.config = defaults.clone();
        }
    }
    Ok(graph)
}

pub fn available_starter_graphs(catalog: &GraphCatalog) -> Vec<GraphDocument> {
    StarterGraph::ALL
        .into_iter()
        .filter_map(|starter| starter_graph(starter, catalog).ok())
        .collect()
}

fn component(
    starter: StarterGraph,
    catalog: &GraphCatalog,
    node_kind: &'static str,
) -> Result<String, StarterError> {
    catalog
        .components
        .values()
        .find(|component| {
            component.node_kind == node_kind && component.readiness == Readiness::Ready
        })
        .map(|component| component.id.clone())
        .ok_or(StarterError::MissingComponent {
            starter: starter.id(),
            node_kind,
        })
}

fn add(graph: &mut GraphDocument, id: &str, kind: &str, component_id: Option<String>) {
    graph.nodes.push(GraphNode {
        id: id.into(),
        kind: kind.into(),
        component_id,
        config: Default::default(),
        disabled: false,
        bypassed: false,
    });
}

fn edge(graph: &mut GraphDocument, from: &str, from_port: &str, to: &str, to_port: &str) {
    graph.edges.push(GraphEdge {
        id: format!("edge:{from}:{from_port}:{to}:{to_port}"),
        from: Endpoint::new(from, from_port),
        to: Endpoint::new(to, to_port),
        capacity: default_edge_capacity(),
    });
}

fn select(graph: &mut GraphDocument, node: &str, port: &str) {
    graph.selected_sinks.push(Endpoint::new(node, port));
}

pub fn fixture_catalog() -> GraphCatalog {
    use crate::{ComponentSpec, Readiness};
    let mut catalog = GraphCatalog::builtin();
    for (id, kind, capability) in [
        ("fixture-asr", "asr", "asr"),
        ("fixture-diarization", "diarization", "diarization"),
        ("fixture-interpretation", "interpretation", "interpretation"),
        ("fixture-response", "response", "text_generation"),
        ("fixture-tts", "tts", "tts"),
    ] {
        catalog.register_component(ComponentSpec {
            id: id.into(),
            node_kind: kind.into(),
            provider: "fixture".into(),
            model: format!("{id}-v1"),
            readiness: Readiness::Ready,
            capabilities: BTreeSet::from([capability.into()]),
            configuration_schema: serde_json::json!({"type":"object"}),
            default_config: serde_json::json!({}),
            detail: "deterministic contract fixture".into(),
            replacement: crate::ReplacementSpec::for_node_kind(kind),
        });
    }
    catalog
}
