use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub const GRAPH_SCHEMA_VERSION: u32 = 3;
pub const PRESENTATION_SCHEMA_VERSION: u32 = 1;

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
    /// Editor organization that never changes execution semantics.
    #[serde(default)]
    pub presentation: GraphPresentation,
    /// Explicit semantic boundaries over ordinary runtime nodes and edges.
    ///
    /// Subpatches are embedded snapshots in schema 3. Their member nodes remain
    /// in `nodes`, so compilation never depends on invisible frontend routing.
    #[serde(default)]
    pub subpatches: Vec<Subpatch>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphPresentation {
    #[serde(default = "default_presentation_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub node_positions: BTreeMap<String, CanvasPoint>,
    #[serde(default)]
    pub node_faceplates: BTreeMap<String, NodeFaceplatePresentation>,
    #[serde(default)]
    pub frames: Vec<PresentationFrame>,
    #[serde(default)]
    pub notes: Vec<PresentationNote>,
    #[serde(default)]
    pub cables: BTreeMap<String, CablePresentation>,
    #[serde(default)]
    pub collapsed_subpatches: Vec<String>,
    #[serde(default = "default_cable_opacity")]
    pub global_cable_opacity: f32,
    #[serde(default)]
    pub selected_path_focus: bool,
}

impl Default for GraphPresentation {
    fn default() -> Self {
        Self {
            schema_version: PRESENTATION_SCHEMA_VERSION,
            node_positions: BTreeMap::new(),
            node_faceplates: BTreeMap::new(),
            frames: Vec::new(),
            notes: Vec::new(),
            cables: BTreeMap::new(),
            collapsed_subpatches: Vec::new(),
            global_cable_opacity: default_cable_opacity(),
            selected_path_focus: false,
        }
    }
}

const fn default_presentation_schema_version() -> u32 {
    PRESENTATION_SCHEMA_VERSION
}

const fn default_cable_opacity() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CanvasPoint {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeFaceplatePresentation {
    #[serde(default)]
    pub collapsed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collapsed_height: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresentationFrame {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub color: String,
    pub origin: CanvasPoint,
    pub size: CanvasPoint,
    #[serde(default)]
    pub node_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresentationNote {
    pub id: String,
    pub text: String,
    pub position: CanvasPoint,
    #[serde(default)]
    pub color: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CableRouting {
    Straight,
    #[default]
    Curved,
    Orthogonal,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CablePresentation {
    #[serde(default)]
    pub routing: CableRouting,
    #[serde(default)]
    pub reroute_points: Vec<CanvasPoint>,
    #[serde(default)]
    pub emphasized: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Subpatch {
    pub id: String,
    pub title: String,
    /// Stable embedded-definition identity retained by duplicates.
    pub definition_id: String,
    #[serde(default = "default_definition_version")]
    pub definition_version: u64,
    #[serde(default)]
    pub parent_subpatch_id: Option<String>,
    #[serde(default)]
    pub node_ids: Vec<String>,
    #[serde(default)]
    pub exposed_ports: Vec<SubpatchPort>,
}

const fn default_definition_version() -> u64 {
    1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubpatchPort {
    pub id: String,
    pub label: String,
    pub direction: crate::PortDirection,
    pub value_type: crate::ValueType,
    pub internal: Endpoint,
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
            presentation: GraphPresentation::default(),
            subpatches: Vec::new(),
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
/// Schema 1 used `id`, `name`, and string sink node IDs. Schema 2 made graph
/// identity and sink ports explicit. Schema 3 promotes presentation organization
/// into a typed contract and adds explicit embedded subpatch boundaries.
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
        2 => migrate_v2(value),
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
            "add versioned presentation metadata and embedded subpatch boundaries".into(),
        ],
        document,
    })
}

fn migrate_v2(mut value: Value) -> Result<MigrationReport, MigrationError> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| MigrationError::InvalidJson("graph must be an object".into()))?;
    object.insert("schema_version".into(), Value::from(GRAPH_SCHEMA_VERSION));
    let mut presentation = serde_json::to_value(GraphPresentation::default())
        .expect("presentation defaults serialize");
    if let Some(labels) = object
        .get("metadata")
        .and_then(|value| value.get("labels"))
        .and_then(Value::as_object)
    {
        let parsed = |key: &str| {
            labels
                .get(key)
                .and_then(Value::as_str)
                .and_then(|value| serde_json::from_str::<Value>(value).ok())
        };
        if let Some(layout) = parsed("studio.layout.v1") {
            presentation["node_positions"] = layout;
        }
        let collapsed = parsed("studio.node-faceplate.v1")
            .and_then(|value| value.get("collapsed").cloned())
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        let geometry = parsed("studio.node-faceplate-geometry.v1")
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();
        let mut faceplates = Map::new();
        for node_id in collapsed.keys().chain(geometry.keys()) {
            let mut faceplate = geometry
                .get(node_id)
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            faceplate.insert(
                "collapsed".into(),
                Value::Bool(collapsed.contains_key(node_id)),
            );
            faceplates.insert(node_id.clone(), Value::Object(faceplate));
        }
        presentation["node_faceplates"] = Value::Object(faceplates);
    }
    object.insert("presentation".into(), presentation);
    object
        .entry("subpatches")
        .or_insert_with(|| Value::Array(Vec::new()));
    let document = serde_json::from_value(value)
        .map_err(|error| MigrationError::InvalidJson(error.to_string()))?;
    Ok(MigrationReport {
        from_version: 2,
        to_version: GRAPH_SCHEMA_VERSION,
        steps: vec![
            "promote layout and faceplate labels into `presentation`".into(),
            "add explicit embedded subpatch boundaries".into(),
        ],
        document,
    })
}
