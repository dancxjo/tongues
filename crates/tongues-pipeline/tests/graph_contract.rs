use serde_json::json;
use tongues_pipeline::*;

#[test]
fn catalog_exposes_backend_owned_replacement_identity_and_mapping_contracts() {
    let mut catalog = fixture_catalog();
    assert_eq!(catalog.schema_version, 2);
    for kind in catalog.node_kinds.values() {
        assert_eq!(kind.replacement.family, kind.kind);
        assert!(!kind.replacement.configuration_schema_id.is_empty());
        assert!(kind.replacement.configuration_schema_version > 0);
    }
    let component = catalog.components.get_mut("fixture-asr").unwrap();
    component
        .replacement
        .port_aliases
        .insert("audio".into(), "samples".into());
    component
        .replacement
        .configuration_aliases
        .insert("language".into(), "locale".into());
    let serialized = serde_json::to_value(&catalog).unwrap();
    assert_eq!(
        serialized["components"]["fixture-asr"]["replacement"]["family"],
        "asr"
    );
    assert_eq!(
        serialized["components"]["fixture-asr"]["replacement"]["port_aliases"]["audio"],
        "samples"
    );
    assert_eq!(
        serialized["components"]["fixture-asr"]["replacement"]["configuration_aliases"]["language"],
        "locale"
    );
}

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
fn text_source_configuration_round_trips_and_executes_verbatim() {
    let catalog = fixture_catalog();
    let mut graph = starter_graph(StarterGraph::TextToSpeech, &catalog).unwrap();
    let source = graph
        .nodes
        .iter_mut()
        .find(|node| node.kind == "text_source")
        .unwrap();
    source.config.insert(
        "text".into(),
        json!("First line.\nSecond line with punctuation: č, ə, ʃ."),
    );
    let serialized = serde_json::to_value(&graph).unwrap();
    let reopened = migrate_graph_json(serialized).unwrap().document;
    let plan = compile_graph(&reopened, &catalog).unwrap();
    let source_step = plan
        .steps
        .iter()
        .find(|step| step.node_kind == "text_source")
        .unwrap();
    let output = configured_source_output(source_step).unwrap();
    assert_eq!(output.port_id, "out");
    assert_eq!(
        output.value,
        json!("First line.\nSecond line with punctuation: č, ə, ʃ.")
    );
    assert!(plan.channels.iter().any(|channel| {
        channel.from == Endpoint::new("text", "out") && channel.to == Endpoint::new("tts", "in")
    }));
}

#[test]
fn text_sources_declare_stream_delivery_and_source_specific_configuration() {
    let catalog = fixture_catalog();
    for (kind, field, format) in [
        ("text_source", "text", "multiline"),
        ("text_file", "path", "path"),
        ("text_url", "url", "uri"),
    ] {
        let source = catalog.node_kinds.get(kind).unwrap();
        let output = source.ports.iter().find(|port| port.id == "out").unwrap();
        assert_eq!(output.value_type, ValueType::Text);
        assert_eq!(output.cardinality, Cardinality::Many);
        assert!(output.streaming, "{kind} must declare stream delivery");
        assert_eq!(
            source.configuration_schema["properties"][field]["format"],
            format
        );
        assert!(
            source.configuration_schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .any(|required| required == field)
        );
    }
}

#[test]
fn phonetic_segmentation_is_typed_and_requires_an_alignment_component() {
    let catalog = GraphCatalog::builtin();
    let node = catalog.node_kinds.get("phonetic_segmentation").unwrap();
    assert!(node.requires_component);
    assert!(node.required_capabilities.contains("phonetic_segmentation"));
    assert!(node.ports.iter().any(|port| {
        port.id == "audio"
            && port.direction == PortDirection::Input
            && port.value_type == ValueType::AudioBuffer
    }));
    assert!(node.ports.iter().any(|port| {
        port.id == "segments"
            && port.direction == PortDirection::Output
            && port.value_type == ValueType::PhoneticSegments
    }));
    assert!(node.ports.iter().any(|port| {
        port.id == "alignment_hints"
            && port.direction == PortDirection::Input
            && port.cardinality == Cardinality::Optional
    }));
}

#[test]
fn phone_alignment_has_distinct_posterior_hypothesis_and_streaming_delta_ports() {
    let catalog = GraphCatalog::builtin();
    let node = catalog.node_kinds.get("phone_alignment").unwrap();
    assert!(node.requires_component);
    assert!(node.required_capabilities.contains("phone_alignment"));
    assert!(node.ports.iter().any(|port| {
        port.id == "acoustic_posteriors"
            && port.direction == PortDirection::Input
            && port.value_type == ValueType::AcousticPosteriors
            && port.cardinality == Cardinality::One
    }));
    assert!(node.ports.iter().any(|port| {
        port.id == "hypotheses"
            && port.direction == PortDirection::Output
            && port.value_type == ValueType::AlignmentHypotheses
    }));
    assert!(node.ports.iter().any(|port| {
        port.id == "deltas"
            && port.direction == PortDirection::Output
            && port.value_type == ValueType::AlignmentDeltas
            && port.streaming
    }));
}

#[test]
fn file_and_url_text_sources_compile_as_text_stream_producers() {
    let catalog = fixture_catalog();
    for (kind, field, value) in [
        ("text_file", "path", "docs/speech-dataflow.md"),
        ("text_url", "url", "https://example.com/source.txt"),
    ] {
        let mut graph = starter_graph(StarterGraph::TextToSpeech, &catalog).unwrap();
        let source = graph
            .nodes
            .iter_mut()
            .find(|node| node.kind == "text_source")
            .unwrap();
        source.kind = kind.into();
        source.config = serde_json::Map::from_iter([(field.into(), json!(value))]);
        let plan = compile_graph(&graph, &catalog).unwrap();
        assert!(plan.steps.iter().any(|step| step.node_kind == kind));
        assert!(plan.channels.iter().any(|channel| {
            channel.from == Endpoint::new("text", "out")
                && channel.value_type == ValueType::Text
                && channel.streaming
        }));
    }
}

#[test]
fn configured_sources_reject_empty_saved_values() {
    let catalog = fixture_catalog();
    for (starter, kind, field) in [
        (StarterGraph::TextToSpeech, "text_source", "text"),
        (StarterGraph::Transcription, "audio_file", "path"),
    ] {
        let mut graph = starter_graph(starter, &catalog).unwrap();
        let node = graph
            .nodes
            .iter_mut()
            .find(|node| {
                if kind == "audio_file" {
                    node.kind == "microphone"
                } else {
                    node.kind == kind
                }
            })
            .unwrap();
        node.kind = kind.into();
        node.config.insert(field.into(), json!(""));
        let report = validate_graph(&graph, &catalog);
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "config.string_too_short"),
            "{kind}: {:#?}",
            report.diagnostics
        );
    }
}

#[test]
fn component_configuration_schema_is_validated() {
    let mut catalog = fixture_catalog();
    let component = catalog.components.get_mut("fixture-tts").unwrap();
    component.configuration_schema = json!({
        "type": "object",
        "properties": {"voice": {"type": "string", "minLength": 1}},
        "required": ["voice"]
    });
    let graph = starter_graph(StarterGraph::TextToSpeech, &catalog).unwrap();
    let report = validate_graph(&graph, &catalog);
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "config.required_field_missing"
            && diagnostic.target.node_id.as_deref() == Some("tts")
    }));
}

#[test]
fn audio_output_declares_explicit_browser_system_and_wav_destinations() {
    let catalog = fixture_catalog();
    let output = catalog.node_kinds.get("audio_output").unwrap();
    assert_eq!(
        output.configuration_schema["properties"]["target"]["enum"],
        json!(["browser", "system", "wav"])
    );
    assert_eq!(output.default_config["target"], "browser");
    assert_eq!(output.default_config["browser_device_id"], "default");
    assert_eq!(output.default_config["system_device_id"], "default");
    assert_eq!(output.default_config["wav_path"], "data/speech-output.wav");
}

#[test]
fn audio_output_rejects_unknown_targets_and_escaping_wav_paths() {
    let catalog = fixture_catalog();
    let mut graph = starter_graph(StarterGraph::TextToSpeech, &catalog).unwrap();
    let output = graph
        .nodes
        .iter_mut()
        .find(|node| node.kind == "audio_output")
        .unwrap();
    output.config.insert("target".into(), json!("telepathy"));
    output
        .config
        .insert("wav_path".into(), json!("../outside.wav"));
    let report = validate_graph(&graph, &catalog);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|item| item.code == "config.value_not_available")
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|item| item.code == "config.path_outside_workspace")
    );
}

#[test]
fn meeting_graph_fans_out_and_has_explicit_join_semantics() {
    let catalog = fixture_catalog();
    let graph = starter_graph(StarterGraph::MeetingTranscription, &catalog).unwrap();
    let plan = compile_graph(&graph, &catalog).unwrap();
    let fanout = plan
        .channels
        .iter()
        .filter(|channel| channel.from == Endpoint::new("vad", "out"))
        .collect::<Vec<_>>();
    assert_eq!(fanout.len(), 2);
    assert!(fanout.iter().all(|channel| channel.capacity == 16));
    assert!(plan.channels.iter().any(|channel| {
        channel.from == Endpoint::new("microphone", "out")
            && channel.to == Endpoint::new("vad", "in")
    }));
    assert!(plan.steps.iter().any(|step| step.node_id == "join"));
}

#[test]
fn tts_accepts_exactly_one_raw_text_or_linguistic_plan_input() {
    let catalog = fixture_catalog();
    let mut graph = starter_graph(StarterGraph::TextToSpeech, &catalog).unwrap();
    graph.nodes.insert(
        1,
        GraphNode {
            id: "linguistic".into(),
            kind: "linguistic".into(),
            component_id: None,
            config: Default::default(),
            disabled: false,
            bypassed: false,
        },
    );
    graph.edges[0] = GraphEdge {
        id: "text-to-linguistic".into(),
        from: Endpoint::new("text", "out"),
        to: Endpoint::new("linguistic", "in"),
        capacity: 16,
    };
    graph.edges.insert(
        1,
        GraphEdge {
            id: "plan-to-tts".into(),
            from: Endpoint::new("linguistic", "out"),
            to: Endpoint::new("tts", "plan"),
            capacity: 16,
        },
    );
    assert!(validate_graph(&graph, &catalog).valid);

    graph.edges.push(GraphEdge {
        id: "raw-text-too".into(),
        from: Endpoint::new("text", "out"),
        to: Endpoint::new("tts", "in"),
        capacity: 16,
    });
    assert!(
        validate_graph(&graph, &catalog)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "port.alternative_input_required")
    );
}

#[test]
fn disabled_and_structurally_bypassed_nodes_do_not_enter_the_execution_plan() {
    let catalog = fixture_catalog();
    let mut graph = starter_graph(StarterGraph::TextToSpeech, &catalog).unwrap();
    graph.nodes.push(GraphNode {
        id: "disabled-control".into(),
        kind: "control_source".into(),
        component_id: None,
        config: Default::default(),
        disabled: true,
        bypassed: false,
    });
    graph.nodes.push(GraphNode {
        id: "bypassed-control".into(),
        kind: "control_source".into(),
        component_id: None,
        config: Default::default(),
        disabled: false,
        bypassed: true,
    });
    let plan = compile_graph(&graph, &catalog).unwrap();
    assert!(
        !plan
            .steps
            .iter()
            .any(|step| step.node_id == "disabled-control" || step.node_id == "bypassed-control")
    );
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
    assert_eq!(report.steps.len(), 4);
    assert_eq!(report.document.presentation.schema_version, 1);
    assert!(report.document.subpatches.is_empty());
}

#[test]
fn v2_presentation_labels_migrate_into_typed_metadata() {
    let report = migrate_graph_json(json!({
        "schema_version": 2,
        "graph_id": "legacy-layout",
        "revision": 2,
        "metadata": {"name":"Layout","labels":{
            "studio.layout.v1":"{\"source\":{\"x\":12,\"y\":34}}",
            "studio.node-faceplate.v1":"{\"collapsed\":{\"source\":true}}",
            "studio.node-faceplate-geometry.v1":"{\"source\":{\"width\":300,\"height\":180,\"collapsed_height\":60}}"
        }},
        "nodes": [], "edges": [], "selected_sinks": []
    })).unwrap();
    assert_eq!(report.to_version, GRAPH_SCHEMA_VERSION);
    assert_eq!(
        report.document.presentation.node_positions["source"].x,
        12.0
    );
    assert!(report.document.presentation.node_faceplates["source"].collapsed);
    assert_eq!(
        report.document.presentation.node_faceplates["source"].width,
        Some(300.0)
    );
}

#[test]
fn subpatch_boundaries_are_reviewed_without_changing_the_execution_plan() {
    let catalog = fixture_catalog();
    let mut graph = starter_graph(StarterGraph::TextToSpeech, &catalog).unwrap();
    let baseline = compile_graph(&graph, &catalog).unwrap();
    graph.subpatches.push(Subpatch {
        id: "voice".into(),
        title: "Voice".into(),
        definition_id: "definition:voice".into(),
        definition_version: 1,
        parent_subpatch_id: None,
        node_ids: vec!["text".into(), "tts".into()],
        exposed_ports: vec![SubpatchPort {
            id: "audio".into(),
            label: "Audio".into(),
            direction: PortDirection::Output,
            value_type: ValueType::AudioStream,
            internal: Endpoint::new("tts", "out"),
        }],
    });
    let organized = compile_graph(&graph, &catalog).unwrap();
    assert_eq!(organized.steps, baseline.steps);
    assert_eq!(organized.channels, baseline.channels);
    graph.subpatches[0].exposed_ports.clear();
    let report = validate_graph(&graph, &catalog);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|item| item.code == "subpatch.boundary_unreviewed")
    );
}

#[test]
fn explicit_splitter_merge_adapter_and_direct_fanout_remain_distinct() {
    let catalog = fixture_catalog();
    assert!(catalog.node_kinds["transcript_splitter"].splitter.is_some());
    assert!(catalog.node_kinds["transcript_splitter"].merge.is_none());
    assert!(catalog.node_kinds["transcript_merge"].merge.is_some());
    assert!(
        catalog.node_kinds["committed_transcript_to_text"]
            .adapter
            .is_some()
    );
    let graph = starter_graph(StarterGraph::MeetingTranscription, &catalog).unwrap();
    let plan = compile_graph(&graph, &catalog).unwrap();
    let direct = plan
        .channels
        .iter()
        .filter(|channel| channel.from == Endpoint::new("vad", "out"))
        .count();
    assert_eq!(
        direct, 2,
        "legal direct fan-out stays represented as two edges"
    );
}

#[test]
fn recursive_or_excessively_nested_subpatches_fail_gracefully() {
    let catalog = fixture_catalog();
    let mut graph = starter_graph(StarterGraph::TextToSpeech, &catalog).unwrap();
    for index in 0..9 {
        graph.subpatches.push(Subpatch {
            id: format!("nested-{index}"),
            title: format!("Nested {index}"),
            definition_id: format!("definition:{index}"),
            definition_version: 1,
            parent_subpatch_id: (index > 0).then(|| format!("nested-{}", index - 1)),
            node_ids: vec!["text".into()],
            exposed_ports: vec![],
        });
    }
    let report = validate_graph(&graph, &catalog);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|item| item.code == "subpatch.depth_limit")
    );
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
