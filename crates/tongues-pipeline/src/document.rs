use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub const GRAPH_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphDocument {
    pub schema_version: u32,
    pub graph_id: String,
    pub revision: u64,
    #[serde(default)]
    pub metadata: GraphMetadata,
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
    #[serde(default)]
    pub edges: Vec<GraphEdge>,
    #[serde(default)]
    pub selected_sinks: Vec<Endpoint>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphMetadata {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub allow_unsafe_execution: bool,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component_id: Option<String>,
    #[serde(default)]
    pub config: Map<String, Value>,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub bypassed: bool,
}

impl GraphNode {
    pub const fn is_active(&self) -> bool {
        !self.disabled && !self.bypassed
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Endpoint {
    pub node_id: String,
    pub port_id: String,
}

impl Endpoint {
    pub fn new(node_id: impl Into<String>, port_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            port_id: port_id.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub from: Endpoint,
    pub to: Endpoint,
    #[serde(default = "default_edge_capacity")]
    pub capacity: usize,
}

pub const fn default_edge_capacity() -> usize {
    16
}

impl GraphDocument {
    pub fn new(graph_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            schema_version: GRAPH_SCHEMA_VERSION,
            graph_id: graph_id.into(),
            revision: 1,
            metadata: GraphMetadata {
                name: name.into(),
                ..GraphMetadata::default()
            },
            nodes: Vec::new(),
            edges: Vec::new(),
            selected_sinks: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationReport {
    pub from_version: u32,
    pub to_version: u32,
    #[serde(default)]
    pub steps: Vec<String>,
    pub document: GraphDocument,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MigrationError {
    #[error("graph JSON is invalid: {0}")]
    InvalidJson(String),
    #[error(
        "graph schema {found} is newer than supported schema {supported}; upgrade Tongues before execution"
    )]
    FutureSchema { found: u32, supported: u32 },
    #[error("graph schema {0} has no registered migration path")]
    NoMigration(u32),
}

/// Migrates serialized saved graphs before resolving runtime components.
///
/// Schema 1 used `id`, `name`, and string sink node IDs. Schema 2 makes graph
/// identity and sink ports explicit. Runtime/model references remain stable IDs
/// and are deliberately resolved later by validation/compilation.
pub fn migrate_graph_json(value: Value) -> Result<MigrationReport, MigrationError> {
    let from_version = value
        .get("schema_version")
        .and_then(Value::as_u64)
        .unwrap_or(1) as u32;
    if from_version > GRAPH_SCHEMA_VERSION {
        return Err(MigrationError::FutureSchema {
            found: from_version,
            supported: GRAPH_SCHEMA_VERSION,
        });
    }
    match from_version {
        GRAPH_SCHEMA_VERSION => {
            let document = serde_json::from_value(value)
                .map_err(|error| MigrationError::InvalidJson(error.to_string()))?;
            Ok(MigrationReport {
                from_version,
                to_version: GRAPH_SCHEMA_VERSION,
                steps: Vec::new(),
                document,
            })
        }
        1 => migrate_v1(value),
        other => Err(MigrationError::NoMigration(other)),
    }
}

fn migrate_v1(mut value: Value) -> Result<MigrationReport, MigrationError> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| MigrationError::InvalidJson("graph must be an object".into()))?;
    object.insert("schema_version".into(), Value::from(GRAPH_SCHEMA_VERSION));
    if !object.contains_key("graph_id") {
        let id = object
            .remove("id")
            .unwrap_or_else(|| Value::String("migrated-graph".into()));
        object.insert("graph_id".into(), id);
    }
    let name = object.remove("name");
    let metadata = object
        .entry("metadata")
        .or_insert_with(|| Value::Object(Map::new()));
    if let (Some(name), Some(metadata)) = (name, metadata.as_object_mut()) {
        metadata.entry("name").or_insert(name);
    }
    let sink_nodes = object.remove("selected_output_sinks");
    if !object.contains_key("selected_sinks") {
        let sinks = sink_nodes
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| value.as_str().map(str::to_owned))
            .map(|node_id| {
                serde_json::json!({
                    "node_id": node_id,
                    "port_id": "out"
                })
            })
            .collect();
        object.insert("selected_sinks".into(), Value::Array(sinks));
    }
    let document = serde_json::from_value(value)
        .map_err(|error| MigrationError::InvalidJson(error.to_string()))?;
    Ok(MigrationReport {
        from_version: 1,
        to_version: GRAPH_SCHEMA_VERSION,
        steps: vec![
            "rename `id` to `graph_id`".into(),
            "move `name` into `metadata.name`".into(),
            "replace selected output node IDs with explicit sink endpoints".into(),
        ],
        document,
    })
}
