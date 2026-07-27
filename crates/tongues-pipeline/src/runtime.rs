use crate::{ExecutionPlan, LifecycleEventKind, PlanStep};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeOutput {
    pub port_id: String,
    pub value: Value,
    #[serde(default)]
    pub derived_from: Vec<String>,
}

/// Returns the value produced directly by a saved, configuration-backed source.
///
/// Live sources such as microphones and control ingress deliberately return
/// `None`: their values must come from a runtime adapter, not from graph JSON.
pub fn configured_source_output(step: &PlanStep) -> Option<RuntimeOutput> {
    match step.node_kind.as_str() {
        "text_source" => step.config.get("text").cloned().map(|value| RuntimeOutput {
            port_id: "out".into(),
            value,
            derived_from: vec![format!("config:{}:text", step.node_id)],
        }),
        "audio_file" => step.config.get("path").cloned().map(|value| RuntimeOutput {
            port_id: "artifact".into(),
            value,
            derived_from: vec![format!("config:{}:path", step.node_id)],
        }),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionEvent {
    pub sequence: u64,
    pub plan_id: String,
    pub graph_id: String,
    pub graph_revision: u64,
    pub node_id: String,
    pub kind: LifecycleEventKind,
    pub elapsed_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<RuntimeOutput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub plan: ExecutionPlan,
    pub started_unix_ms: u64,
    pub completed_unix_ms: u64,
    pub cancelled: bool,
    pub events: Vec<ExecutionEvent>,
}

pub trait NodeRunner {
    type Error: std::fmt::Display;

    /// Runs a registry-resolved step. Implementations own actual model/provider
    /// resources and return only directly produced values with derivation IDs.
    fn run(
        &mut self,
        step: &PlanStep,
        cancellation: &CancellationToken,
    ) -> Result<Vec<RuntimeOutput>, Self::Error>;

    fn cancel(&mut self, _step: &PlanStep) {}
}

#[derive(Debug, thiserror::Error)]
#[error("node `{node_id}` failed: {detail}")]
pub struct ExecutionError {
    pub node_id: String,
    pub detail: String,
    pub record: Box<ExecutionRecord>,
}

pub fn execute_plan<R: NodeRunner>(
    plan: ExecutionPlan,
    runner: &mut R,
    cancellation: CancellationToken,
) -> Result<ExecutionRecord, ExecutionError> {
    let started_unix_ms = unix_ms();
    let started = std::time::Instant::now();
    let mut events = Vec::new();
    let mut sequence = 0;
    for step in &plan.steps {
        if cancellation.is_cancelled() {
            cancel_remaining(
                &plan,
                runner,
                step.index,
                started,
                &mut sequence,
                &mut events,
            );
            return Ok(record(plan, started_unix_ms, true, events));
        }
        push_event(
            &plan,
            step,
            LifecycleEventKind::Started,
            started,
            &mut sequence,
            &mut events,
            None,
            None,
        );
        match runner.run(step, &cancellation) {
            Ok(outputs) => {
                for output in outputs {
                    push_event(
                        &plan,
                        step,
                        LifecycleEventKind::Output,
                        started,
                        &mut sequence,
                        &mut events,
                        Some(output),
                        None,
                    );
                }
                if cancellation.is_cancelled() {
                    runner.cancel(step);
                    push_event(
                        &plan,
                        step,
                        LifecycleEventKind::Cancelled,
                        started,
                        &mut sequence,
                        &mut events,
                        None,
                        Some("cancellation observed after node output".into()),
                    );
                    cancel_remaining(
                        &plan,
                        runner,
                        step.index + 1,
                        started,
                        &mut sequence,
                        &mut events,
                    );
                    return Ok(record(plan, started_unix_ms, true, events));
                }
                push_event(
                    &plan,
                    step,
                    LifecycleEventKind::Completed,
                    started,
                    &mut sequence,
                    &mut events,
                    None,
                    None,
                );
            }
            Err(error) => {
                let detail = error.to_string();
                push_event(
                    &plan,
                    step,
                    LifecycleEventKind::Failed,
                    started,
                    &mut sequence,
                    &mut events,
                    None,
                    Some(detail.clone()),
                );
                let node_id = step.node_id.clone();
                return Err(ExecutionError {
                    node_id,
                    detail,
                    record: Box::new(record(plan, started_unix_ms, false, events)),
                });
            }
        }
    }
    Ok(record(plan, started_unix_ms, false, events))
}

fn cancel_remaining<R: NodeRunner>(
    plan: &ExecutionPlan,
    runner: &mut R,
    from_index: usize,
    started: std::time::Instant,
    sequence: &mut u64,
    events: &mut Vec<ExecutionEvent>,
) {
    for step in plan.steps.iter().skip(from_index) {
        runner.cancel(step);
        push_event(
            plan,
            step,
            LifecycleEventKind::Cancelled,
            started,
            sequence,
            events,
            None,
            Some("cancelled before start; owned channels closed".into()),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_event(
    plan: &ExecutionPlan,
    step: &PlanStep,
    kind: LifecycleEventKind,
    started: std::time::Instant,
    sequence: &mut u64,
    events: &mut Vec<ExecutionEvent>,
    output: Option<RuntimeOutput>,
    detail: Option<String>,
) {
    *sequence += 1;
    events.push(ExecutionEvent {
        sequence: *sequence,
        plan_id: plan.plan_id.clone(),
        graph_id: plan.graph_id.clone(),
        graph_revision: plan.graph_revision,
        node_id: step.node_id.clone(),
        kind,
        elapsed_ms: started.elapsed().as_millis() as u64,
        component_id: step.component_id.clone(),
        output,
        detail,
    });
}

fn record(
    plan: ExecutionPlan,
    started_unix_ms: u64,
    cancelled: bool,
    events: Vec<ExecutionEvent>,
) -> ExecutionRecord {
    ExecutionRecord {
        plan,
        started_unix_ms,
        completed_unix_ms: unix_ms(),
        cancelled,
        events,
    }
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
