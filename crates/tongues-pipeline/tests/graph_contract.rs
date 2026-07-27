use serde_json::json;
use tongues_pipeline::*;

#[test]
fn starter_graphs_validate_and_compile_deterministically() {
    let catalog = fixture_catalog();
    for starter in StarterGraph::ALL {
        let graph = starter_graph(starter, &catalog).unwrap();
        let report = validate_graph(&graph, &catalog);
        assert!(report.valid, "{starter:?}: {:#?}", report.diagnostics);
        let first = compile_graph(&graph, &catalog).unwrap();
        let second = compile_graph(&graph, &catalog).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.graph_revision, graph.revision);
        assert!(!first.provenance.derivations.is_empty());
    }
}

#[test]
fn meeting_graph_fans_out_and_has_explicit_join_semantics() {
    let catalog = fixture_catalog();
    let graph = starter_graph(StarterGraph::MeetingTranscription, &catalog).unwrap();
    let plan = compile_graph(&graph, &catalog).unwrap();
    let fanout = plan
        .channels
        .iter()
        .filter(|channel| channel.from == Endpoint::new("microphone", "out"))
        .collect::<Vec<_>>();
    assert_eq!(fanout.len(), 2);
    assert!(fanout.iter().all(|channel| channel.capacity == 16));
    assert!(plan.steps.iter().any(|step| step.node_id == "join"));
}

#[test]
fn diagnostics_cover_common_invalid_wiring_cases() {
    let catalog = fixture_catalog();
    let mut graph = starter_graph(StarterGraph::Transcription, &catalog).unwrap();
    graph.edges.retain(|edge| edge.to.node_id != "asr");
    graph.edges.push(GraphEdge {
        id: "bad-type".into(),
        from: Endpoint::new("microphone", "out"),
        to: Endpoint::new("transcript", "in"),
        capacity: 0,
    });
    graph.nodes[1].component_id = Some("missing-model".into());
    graph.edges.push(GraphEdge {
        id: "cycle".into(),
        from: Endpoint::new("asr", "committed"),
        to: Endpoint::new("asr", "audio"),
        capacity: 1,
    });
    let report = validate_graph(&graph, &catalog);
    let codes = report
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.code.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(codes.contains("port.required_input_missing"));
    assert!(codes.contains("edge.incompatible_type"));
    assert!(codes.contains("edge.unbounded_or_zero_capacity"));
    assert!(codes.contains("component.unavailable"));
    assert!(codes.contains("graph.cycle"));
    let incompatible = report
        .diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.code == "edge.incompatible_type"
                && diagnostic.target.edge_id.as_deref() == Some("bad-type")
        })
        .unwrap();
    assert_eq!(incompatible.target.edge_id.as_deref(), Some("bad-type"));
}

#[test]
fn explicit_adapter_is_required_for_buffered_audio() {
    let catalog = fixture_catalog();
    let mut graph = starter_graph(StarterGraph::Transcription, &catalog).unwrap();
    graph.nodes[0].kind = "audio_file".into();
    let report = validate_graph(&graph, &catalog);
    let mismatch = report
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "edge.incompatible_type")
        .unwrap();
    assert_eq!(
        mismatch.suggestions,
        vec!["audio_buffer_to_stream".to_string()]
    );
}

#[test]
fn v1_saved_graph_migrates_with_precise_steps() {
    let report = migrate_graph_json(json!({
        "schema_version": 1,
        "id": "legacy",
        "revision": 7,
        "name": "Legacy graph",
        "nodes": [],
        "edges": [],
        "selected_output_sinks": ["output"]
    }))
    .unwrap();
    assert_eq!(report.document.graph_id, "legacy");
    assert_eq!(report.document.metadata.name, "Legacy graph");
    assert_eq!(
        report.document.selected_sinks,
        vec![Endpoint::new("output", "out")]
    );
    assert_eq!(report.steps.len(), 3);
}

struct CancellingRunner {
    token: CancellationToken,
    calls: usize,
    cancelled_nodes: Vec<String>,
}

impl NodeRunner for CancellingRunner {
    type Error = String;

    fn run(
        &mut self,
        step: &PlanStep,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<RuntimeOutput>, Self::Error> {
        self.calls += 1;
        self.token.cancel();
        Ok(vec![RuntimeOutput {
            port_id: "out".into(),
            value: json!({"node": step.node_id}),
            derived_from: vec!["fixture-input".into()],
        }])
    }

    fn cancel(&mut self, step: &PlanStep) {
        self.cancelled_nodes.push(step.node_id.clone());
    }
}

#[test]
fn cancelled_streaming_run_closes_every_remaining_owner() {
    let catalog = fixture_catalog();
    let graph = starter_graph(StarterGraph::LiveConversation, &catalog).unwrap();
    let plan = compile_graph(&graph, &catalog).unwrap();
    let token = CancellationToken::default();
    let mut runner = CancellingRunner {
        token: token.clone(),
        calls: 0,
        cancelled_nodes: Vec::new(),
    };
    let record = execute_plan(plan.clone(), &mut runner, token).unwrap();
    assert!(record.cancelled);
    assert_eq!(runner.calls, 1);
    assert_eq!(runner.cancelled_nodes.len(), plan.steps.len());
    assert_eq!(
        record
            .events
            .iter()
            .filter(|event| event.kind == LifecycleEventKind::Cancelled)
            .count(),
        plan.steps.len()
    );
}
