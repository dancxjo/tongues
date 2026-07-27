use crate::{
    Cardinality, ComponentSpec, Endpoint, GRAPH_SCHEMA_VERSION, GraphCatalog, GraphDocument,
    GraphEdge, GraphNode, PortDirection, Readiness, ValueType,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticTarget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphDiagnostic {
    pub code: String,
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub target: DiagnosticTarget,
    #[serde(default)]
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationReport {
    pub valid: bool,
    pub graph_id: String,
    pub graph_revision: u64,
    #[serde(default)]
    pub diagnostics: Vec<GraphDiagnostic>,
}

fn target(
    graph: &GraphDocument,
    node: Option<&str>,
    port: Option<&str>,
    edge: Option<&str>,
) -> DiagnosticTarget {
    DiagnosticTarget {
        graph_id: Some(graph.graph_id.clone()),
        node_id: node.map(str::to_owned),
        port_id: port.map(str::to_owned),
        edge_id: edge.map(str::to_owned),
    }
}

fn error(
    graph: &GraphDocument,
    code: &str,
    message: impl Into<String>,
    node: Option<&str>,
    port: Option<&str>,
    edge: Option<&str>,
) -> GraphDiagnostic {
    GraphDiagnostic {
        code: code.into(),
        severity: DiagnosticSeverity::Error,
        message: message.into(),
        target: target(graph, node, port, edge),
        suggestions: Vec::new(),
    }
}

pub fn validate_graph(graph: &GraphDocument, catalog: &GraphCatalog) -> ValidationReport {
    let mut diagnostics = Vec::new();
    if graph.schema_version != GRAPH_SCHEMA_VERSION {
        diagnostics.push(error(
            graph,
            "schema.unsupported",
            format!(
                "Graph schema {} cannot execute; migrate it to schema {}.",
                graph.schema_version, GRAPH_SCHEMA_VERSION
            ),
            None,
            None,
            None,
        ));
    }
    let nodes = unique_nodes(graph, &mut diagnostics);
    let mut edge_ids = BTreeSet::new();
    let mut incoming: BTreeMap<Endpoint, Vec<&GraphEdge>> = BTreeMap::new();
    let mut outgoing: BTreeMap<Endpoint, Vec<&GraphEdge>> = BTreeMap::new();

    for edge in &graph.edges {
        if !edge_ids.insert(edge.id.as_str()) {
            diagnostics.push(error(
                graph,
                "edge.duplicate_id",
                format!("Edge ID `{}` is used more than once.", edge.id),
                None,
                None,
                Some(&edge.id),
            ));
        }
        if edge.capacity == 0 {
            diagnostics.push(error(
                graph,
                "edge.unbounded_or_zero_capacity",
                "Streaming edges need a positive bounded capacity.",
                None,
                None,
                Some(&edge.id),
            ));
        }
        let inactive_endpoint = nodes
            .get(edge.from.node_id.as_str())
            .into_iter()
            .chain(nodes.get(edge.to.node_id.as_str()))
            .find(|node| !node.is_active());
        if let Some(node) = inactive_endpoint {
            diagnostics.push(error(
                graph,
                "edge.inactive_node",
                format!(
                    "Edge `{}` references disabled or bypassed node `{}`; remove the edge or enable the node.",
                    edge.id, node.id
                ),
                Some(&node.id),
                None,
                Some(&edge.id),
            ));
            continue;
        }
        let from = resolved_port(
            graph,
            catalog,
            &nodes,
            &edge.from,
            Some(edge),
            &mut diagnostics,
        );
        let to = resolved_port(
            graph,
            catalog,
            &nodes,
            &edge.to,
            Some(edge),
            &mut diagnostics,
        );
        if let (Some(from), Some(to)) = (from, to) {
            let mut compatible = true;
            if from.direction != PortDirection::Output || to.direction != PortDirection::Input {
                compatible = false;
                diagnostics.push(error(
                    graph,
                    "edge.direction",
                    format!(
                        "Edge `{}` must connect an output port to an input port.",
                        edge.id
                    ),
                    Some(&edge.to.node_id),
                    Some(&edge.to.port_id),
                    Some(&edge.id),
                ));
            } else if from.value_type != to.value_type {
                compatible = false;
                let mut diagnostic = error(
                    graph,
                    "edge.incompatible_type",
                    format!(
                        "`{}.{}` emits {:?}, but `{}.{}` accepts {:?}; conversions must be explicit.",
                        edge.from.node_id,
                        edge.from.port_id,
                        from.value_type,
                        edge.to.node_id,
                        edge.to.port_id,
                        to.value_type
                    ),
                    Some(&edge.to.node_id),
                    Some(&edge.to.port_id),
                    Some(&edge.id),
                );
                diagnostic.suggestions = catalog.adapters_for(from.value_type, to.value_type);
                diagnostics.push(diagnostic);
            }
            if compatible {
                incoming.entry(edge.to.clone()).or_default().push(edge);
                outgoing.entry(edge.from.clone()).or_default().push(edge);
            }
        }
    }

    for node in nodes.values() {
        if !node.is_active() {
            continue;
        }
        let Some(kind) = catalog.node_kinds.get(&node.kind) else {
            diagnostics.push(error(
                graph,
                "node.unknown_kind",
                format!(
                    "Node `{}` references unknown kind `{}`; replace it with a catalog node kind.",
                    node.id, node.kind
                ),
                Some(&node.id),
                None,
                None,
            ));
            continue;
        };
        validate_component(graph, node, kind, catalog, &mut diagnostics);
        let configuration_schema = node
            .component_id
            .as_ref()
            .and_then(|component_id| catalog.components.get(component_id))
            .map_or(&kind.configuration_schema, |component| {
                &component.configuration_schema
            });
        validate_config(graph, node, configuration_schema, &mut diagnostics);
        if kind.unsafe_execution && !graph.metadata.allow_unsafe_execution {
            diagnostics.push(error(
                graph,
                "node.unsafe_execution",
                format!(
                    "Node `{}` requires explicit unsafe-execution approval.",
                    node.id
                ),
                Some(&node.id),
                None,
                None,
            ));
        }
        for port in &kind.ports {
            let endpoint = Endpoint::new(&node.id, &port.id);
            let edges = match port.direction {
                PortDirection::Input => incoming.get(&endpoint),
                PortDirection::Output => outgoing.get(&endpoint),
            };
            let count = edges.map_or(0, Vec::len);
            if port.direction == PortDirection::Input
                && port.cardinality == Cardinality::One
                && count == 0
            {
                diagnostics.push(error(
                    graph,
                    "port.required_input_missing",
                    format!(
                        "`{}.{}` requires exactly one incoming relationship.",
                        node.id, port.id
                    ),
                    Some(&node.id),
                    Some(&port.id),
                    None,
                ));
            }
            if port.cardinality != Cardinality::Many && count > 1 {
                diagnostics.push(error(
                    graph,
                    "port.cardinality",
                    format!(
                        "`{}.{}` accepts at most one relationship, but has {count}. Use an explicit merge node.",
                        node.id, port.id
                    ),
                    Some(&node.id),
                    Some(&port.id),
                    None,
                ));
            }
            if port.direction == PortDirection::Output && port.must_consume && count == 0 {
                diagnostics.push(error(
                    graph,
                    "port.required_output_unconsumed",
                    format!(
                        "`{}.{}` is a required output and must be connected to a consumer.",
                        node.id, port.id
                    ),
                    Some(&node.id),
                    Some(&port.id),
                    None,
                ));
            }
        }
        if kind.kind == "tts" {
            let text = incoming
                .get(&Endpoint::new(&node.id, "in"))
                .map_or(0, Vec::len);
            let plan = incoming
                .get(&Endpoint::new(&node.id, "plan"))
                .map_or(0, Vec::len);
            if text + plan != 1 {
                let mut diagnostic = error(
                    graph,
                    "port.alternative_input_required",
                    format!(
                        "`{}` requires exactly one speech input: raw text on `in` or a linguistic/prosody utterance plan on `plan`.",
                        node.id
                    ),
                    Some(&node.id),
                    Some("in"),
                    None,
                );
                diagnostic.suggestions.push(
                    "Connect a text source to `in`, or connect linguistic analysis to `plan`."
                        .into(),
                );
                diagnostics.push(diagnostic);
            }
        }
    }
    validate_cycles(graph, catalog, &nodes, &mut diagnostics);
    validate_sinks(graph, catalog, &nodes, &incoming, &mut diagnostics);
    diagnostics.sort_by(|a, b| {
        (
            &a.target.node_id,
            &a.target.port_id,
            &a.target.edge_id,
            &a.code,
        )
            .cmp(&(
                &b.target.node_id,
                &b.target.port_id,
                &b.target.edge_id,
                &b.code,
            ))
    });
    ValidationReport {
        valid: diagnostics
            .iter()
            .all(|item| item.severity != DiagnosticSeverity::Error),
        graph_id: graph.graph_id.clone(),
        graph_revision: graph.revision,
        diagnostics,
    }
}

fn unique_nodes<'a>(
    graph: &'a GraphDocument,
    diagnostics: &mut Vec<GraphDiagnostic>,
) -> BTreeMap<&'a str, &'a GraphNode> {
    let mut nodes = BTreeMap::new();
    for node in &graph.nodes {
        if nodes.insert(node.id.as_str(), node).is_some() {
            diagnostics.push(error(
                graph,
                "node.duplicate_id",
                format!("Node ID `{}` is used more than once.", node.id),
                Some(&node.id),
                None,
                None,
            ));
        }
    }
    nodes
}

fn resolved_port<'a>(
    graph: &GraphDocument,
    catalog: &'a GraphCatalog,
    nodes: &BTreeMap<&str, &GraphNode>,
    endpoint: &Endpoint,
    edge: Option<&GraphEdge>,
    diagnostics: &mut Vec<GraphDiagnostic>,
) -> Option<&'a crate::PortSpec> {
    let Some(node) = nodes.get(endpoint.node_id.as_str()) else {
        diagnostics.push(error(
            graph,
            "edge.node_missing",
            format!("Endpoint references missing node `{}`.", endpoint.node_id),
            Some(&endpoint.node_id),
            Some(&endpoint.port_id),
            edge.map(|edge| edge.id.as_str()),
        ));
        return None;
    };
    let kind = catalog.node_kinds.get(&node.kind)?;
    let port = kind.ports.iter().find(|port| port.id == endpoint.port_id);
    if port.is_none() {
        diagnostics.push(error(
            graph,
            "edge.port_missing",
            format!(
                "Node `{}` has no `{}` port in current catalog revision `{}`.",
                endpoint.node_id, endpoint.port_id, catalog.revision
            ),
            Some(&endpoint.node_id),
            Some(&endpoint.port_id),
            edge.map(|edge| edge.id.as_str()),
        ));
    }
    port
}

fn validate_component(
    graph: &GraphDocument,
    node: &GraphNode,
    kind: &crate::NodeKindSpec,
    catalog: &GraphCatalog,
    diagnostics: &mut Vec<GraphDiagnostic>,
) {
    if !kind.requires_component && node.component_id.is_none() {
        return;
    }
    let Some(component_id) = &node.component_id else {
        diagnostics.push(error(
            graph,
            "component.required",
            format!("Node `{}` needs a backend component selection.", node.id),
            Some(&node.id),
            None,
            None,
        ));
        return;
    };
    let Some(component) = catalog.components.get(component_id) else {
        diagnostics.push(error(
            graph,
            "component.unavailable",
            format!(
                "Component `{component_id}` is not present in current backend discovery; choose an available replacement."
            ),
            Some(&node.id),
            None,
            None,
        ));
        return;
    };
    if component.node_kind != node.kind {
        diagnostics.push(error(
            graph,
            "component.wrong_kind",
            format!(
                "Component `{component_id}` belongs to node kind `{}`, not `{}`.",
                component.node_kind, node.kind
            ),
            Some(&node.id),
            None,
            None,
        ));
    }
    if component.readiness != Readiness::Ready {
        diagnostics.push(error(
            graph,
            "component.not_ready",
            format!(
                "Component `{component_id}` is {:?}: {}",
                component.readiness, component.detail
            ),
            Some(&node.id),
            None,
            None,
        ));
    }
    for capability in &kind.required_capabilities {
        if !component.capabilities.contains(capability) {
            diagnostics.push(error(
                graph,
                "component.capability_missing",
                format!(
                    "Component `{component_id}` does not advertise required capability `{capability}`."
                ),
                Some(&node.id),
                None,
                None,
            ));
        }
    }
}

fn validate_config(
    graph: &GraphDocument,
    node: &GraphNode,
    schema: &serde_json::Value,
    diagnostics: &mut Vec<GraphDiagnostic>,
) {
    let required = schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .collect::<BTreeSet<_>>();
    let properties = schema
        .get("properties")
        .and_then(serde_json::Value::as_object);
    for field in &required {
        if !node.config.contains_key(*field) {
            diagnostics.push(error(
                graph,
                "config.required_field_missing",
                format!("Node `{}` configuration requires `{field}`.", node.id),
                Some(&node.id),
                None,
                None,
            ));
        }
    }
    let Some(properties) = properties else {
        return;
    };
    for (field, field_schema) in properties {
        let Some(value) = node.config.get(field) else {
            continue;
        };
        let expected = field_schema.get("type").and_then(serde_json::Value::as_str);
        let type_matches = match expected {
            Some("string") => value.is_string(),
            Some("number") => value.is_number(),
            Some("integer") => value.as_i64().is_some() || value.as_u64().is_some(),
            Some("boolean") => value.is_boolean(),
            Some("array") => value.is_array(),
            Some("object") => value.is_object(),
            _ => true,
        };
        if !type_matches {
            diagnostics.push(error(
                graph,
                "config.invalid_type",
                format!(
                    "Node `{}` configuration field `{field}` must be {}.",
                    node.id,
                    expected.unwrap_or("a supported value")
                ),
                Some(&node.id),
                None,
                None,
            ));
            continue;
        }
        if let (Some(text), Some(min_length)) = (
            value.as_str(),
            field_schema
                .get("minLength")
                .and_then(serde_json::Value::as_u64),
        ) && text.chars().count() < min_length as usize
        {
            diagnostics.push(error(
                graph,
                "config.string_too_short",
                format!(
                    "Node `{}` configuration field `{field}` must contain at least {min_length} character{}.",
                    node.id,
                    if min_length == 1 { "" } else { "s" }
                ),
                Some(&node.id),
                None,
                None,
            ));
        }
    }
}

fn validate_cycles(
    graph: &GraphDocument,
    catalog: &GraphCatalog,
    nodes: &BTreeMap<&str, &GraphNode>,
    diagnostics: &mut Vec<GraphDiagnostic>,
) {
    let mut indegree = nodes
        .keys()
        .map(|id| (*id, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut outgoing = BTreeMap::<&str, Vec<&str>>::new();
    for edge in &graph.edges {
        if nodes.contains_key(edge.from.node_id.as_str())
            && nodes.contains_key(edge.to.node_id.as_str())
        {
            *indegree.entry(&edge.to.node_id).or_default() += 1;
            outgoing
                .entry(&edge.from.node_id)
                .or_default()
                .push(&edge.to.node_id);
        }
    }
    let mut queue = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
        .collect::<VecDeque<_>>();
    while let Some(node) = queue.pop_front() {
        for child in outgoing.get(node).into_iter().flatten() {
            let degree = indegree.get_mut(child).expect("known graph node");
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(child);
            }
        }
    }
    let cycle = indegree
        .into_iter()
        .filter_map(|(id, degree)| (degree > 0).then_some(id))
        .filter(|id| {
            nodes
                .get(id)
                .and_then(|node| catalog.node_kinds.get(&node.kind))
                .is_none_or(|kind| !kind.permits_cycle)
        })
        .collect::<Vec<_>>();
    if !cycle.is_empty() {
        diagnostics.push(error(
            graph,
            "graph.cycle",
            format!(
                "Graph contains an execution cycle through [{}]; insert an explicitly cycle-safe control node or remove the cycle.",
                cycle.join(", ")
            ),
            None,
            None,
            None,
        ));
    }
}

fn validate_sinks(
    graph: &GraphDocument,
    catalog: &GraphCatalog,
    nodes: &BTreeMap<&str, &GraphNode>,
    incoming: &BTreeMap<Endpoint, Vec<&GraphEdge>>,
    diagnostics: &mut Vec<GraphDiagnostic>,
) {
    if graph.selected_sinks.is_empty() {
        diagnostics.push(error(
            graph,
            "graph.sink_missing",
            "Select at least one output sink.",
            None,
            None,
            None,
        ));
    }
    for sink in &graph.selected_sinks {
        let Some(node) = nodes.get(sink.node_id.as_str()) else {
            diagnostics.push(error(
                graph,
                "graph.sink_node_missing",
                format!("Selected sink node `{}` no longer exists.", sink.node_id),
                Some(&sink.node_id),
                Some(&sink.port_id),
                None,
            ));
            continue;
        };
        let exists = catalog
            .node_kinds
            .get(&node.kind)
            .is_some_and(|kind| kind.ports.iter().any(|port| port.id == sink.port_id));
        if !exists {
            diagnostics.push(error(
                graph,
                "graph.sink_port_missing",
                format!(
                    "Selected sink `{}.{}` no longer exists in the current catalog.",
                    sink.node_id, sink.port_id
                ),
                Some(&sink.node_id),
                Some(&sink.port_id),
                None,
            ));
        } else if !incoming.contains_key(sink) {
            diagnostics.push(error(
                graph,
                "graph.sink_disconnected",
                format!(
                    "Selected sink `{}.{}` is disconnected from pipeline output.",
                    sink.node_id, sink.port_id
                ),
                Some(&sink.node_id),
                Some(&sink.port_id),
                None,
            ));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub schema_version: u32,
    pub plan_id: String,
    pub graph_id: String,
    pub graph_revision: u64,
    pub catalog_revision: String,
    pub steps: Vec<PlanStep>,
    pub channels: Vec<PlanChannel>,
    pub selected_sinks: Vec<Endpoint>,
    pub cancellation: CancellationPlan,
    pub provenance: PlanProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStep {
    pub index: usize,
    pub node_id: String,
    pub node_kind: String,
    pub component_id: Option<String>,
    pub config: serde_json::Map<String, serde_json::Value>,
    pub resource_owner: String,
    pub lifecycle: Vec<LifecycleEventKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge: Option<crate::MergeSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanChannel {
    pub edge_id: String,
    pub from: Endpoint,
    pub to: Endpoint,
    pub value_type: ValueType,
    #[serde(default)]
    pub streaming: bool,
    pub capacity: usize,
    pub backpressure: String,
    pub producer_owns_close: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancellationPlan {
    pub propagation: String,
    pub affected_nodes: Vec<String>,
    pub closes_all_channels: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanProvenance {
    pub graph_id: String,
    pub graph_revision: u64,
    pub catalog_revision: String,
    pub resolved_components: Vec<ResolvedComponent>,
    pub derivations: Vec<Derivation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedComponent {
    pub node_id: String,
    pub component_id: String,
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Derivation {
    pub edge_id: String,
    pub source: Endpoint,
    pub output: Endpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleEventKind {
    Planned,
    Started,
    Output,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompileFailure {
    pub validation: ValidationReport,
}

pub fn compile_graph(
    graph: &GraphDocument,
    catalog: &GraphCatalog,
) -> Result<ExecutionPlan, CompileFailure> {
    let validation = validate_graph(graph, catalog);
    if !validation.valid {
        return Err(CompileFailure { validation });
    }
    let order = topological_order(graph);
    let steps = order
        .iter()
        .enumerate()
        .map(|(index, node_id)| {
            let node = graph
                .nodes
                .iter()
                .find(|node| node.id == *node_id)
                .expect("validated node");
            let kind = &catalog.node_kinds[&node.kind];
            PlanStep {
                index,
                node_id: node.id.clone(),
                node_kind: node.kind.clone(),
                component_id: node.component_id.clone(),
                config: node.config.clone(),
                resource_owner: format!("node:{}", node.id),
                lifecycle: vec![
                    LifecycleEventKind::Planned,
                    LifecycleEventKind::Started,
                    LifecycleEventKind::Output,
                    LifecycleEventKind::Completed,
                    LifecycleEventKind::Cancelled,
                    LifecycleEventKind::Failed,
                ],
                merge: kind.merge.clone(),
            }
        })
        .collect::<Vec<_>>();
    let channels = graph
        .edges
        .iter()
        .map(|edge| {
            let node = graph
                .nodes
                .iter()
                .find(|node| node.id == edge.from.node_id)
                .expect("validated source");
            let kind = &catalog.node_kinds[&node.kind];
            let port = kind
                .ports
                .iter()
                .find(|port| port.id == edge.from.port_id)
                .expect("validated port");
            PlanChannel {
                edge_id: edge.id.clone(),
                from: edge.from.clone(),
                to: edge.to.clone(),
                value_type: port.value_type,
                streaming: port.streaming,
                capacity: edge.capacity,
                backpressure: "block_producer_until_capacity_or_cancellation".into(),
                producer_owns_close: true,
            }
        })
        .collect::<Vec<_>>();
    let resolved_components = graph
        .nodes
        .iter()
        .filter(|node| node.is_active())
        .filter_map(|node| {
            node.component_id.as_ref().map(|component_id| {
                let component: &ComponentSpec = &catalog.components[component_id];
                ResolvedComponent {
                    node_id: node.id.clone(),
                    component_id: component_id.clone(),
                    provider: component.provider.clone(),
                    model: component.model.clone(),
                }
            })
        })
        .collect();
    let derivations = graph
        .edges
        .iter()
        .map(|edge| Derivation {
            edge_id: edge.id.clone(),
            source: edge.from.clone(),
            output: edge.to.clone(),
        })
        .collect();
    Ok(ExecutionPlan {
        schema_version: 1,
        plan_id: format!("{}@{}:{}", graph.graph_id, graph.revision, catalog.revision),
        graph_id: graph.graph_id.clone(),
        graph_revision: graph.revision,
        catalog_revision: catalog.revision.clone(),
        steps,
        channels,
        selected_sinks: graph.selected_sinks.clone(),
        cancellation: CancellationPlan {
            propagation: "downstream_and_upstream_resource_owners".into(),
            affected_nodes: order,
            closes_all_channels: true,
        },
        provenance: PlanProvenance {
            graph_id: graph.graph_id.clone(),
            graph_revision: graph.revision,
            catalog_revision: catalog.revision.clone(),
            resolved_components,
            derivations,
        },
    })
}

fn topological_order(graph: &GraphDocument) -> Vec<String> {
    let mut indegree = graph
        .nodes
        .iter()
        .filter(|node| node.is_active())
        .map(|node| (node.id.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut outgoing = BTreeMap::<String, Vec<String>>::new();
    for edge in &graph.edges {
        *indegree.entry(edge.to.node_id.clone()).or_default() += 1;
        outgoing
            .entry(edge.from.node_id.clone())
            .or_default()
            .push(edge.to.node_id.clone());
    }
    for children in outgoing.values_mut() {
        children.sort();
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(node.clone()))
        .collect::<BTreeSet<_>>();
    let mut order = Vec::new();
    while let Some(node) = ready.pop_first() {
        order.push(node.clone());
        for child in outgoing.get(&node).into_iter().flatten() {
            let degree = indegree.get_mut(child).expect("known child");
            *degree -= 1;
            if *degree == 0 {
                ready.insert(child.clone());
            }
        }
    }
    order
}
