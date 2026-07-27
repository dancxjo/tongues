use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::data::variety_by_code;
use crate::ids::VarietyId;
use crate::syntax::{GrammarBackend, GrammarParserBackend};
use crate::syntax_link_grammar::link_grammar_readiness;

pub const DEFAULT_UDPIPE_TIMEOUT_MS: u64 = 2_000;
pub const DEFAULT_UDPIPE_MAX_INPUT_BYTES: usize = 64 * 1024;
pub const DEFAULT_UDPIPE_MAX_STDOUT_BYTES: usize = 2 * 1024 * 1024;
pub const DEFAULT_UDPIPE_MAX_STDERR_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarBackendReport {
    pub requested: GrammarParserBackend,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected: Option<GrammarBackend>,
    #[serde(default)]
    pub attempts: Vec<GrammarBackendAttempt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<GrammarFallbackReason>,
}

impl Default for GrammarBackendReport {
    fn default() -> Self {
        Self {
            requested: GrammarParserBackend::Auto,
            selected: None,
            attempts: Vec::new(),
            fallback_reason: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarBackendAttempt {
    pub backend: GrammarBackend,
    pub state: GrammarBackendState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<GrammarBackendIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<GrammarProjectionReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<GrammarCoverageReport>,
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
}

impl GrammarBackendAttempt {
    pub fn native(
        state: GrammarBackendState,
        diagnostic: Option<String>,
        duration_ms: u64,
    ) -> Self {
        Self {
            backend: GrammarBackend::TonguesRules,
            state,
            diagnostic,
            identity: Some(GrammarBackendIdentity::native()),
            projection: None,
            coverage: None,
            duration_ms,
            exit_code: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarBackendState {
    Ready,
    FeatureDisabled,
    UnsupportedVariety,
    UnavailableExecutable,
    UnavailableDictionary,
    UnavailableModel,
    SpawnFailure,
    Timeout,
    Cancelled,
    MalformedOutput,
    InputTooLarge,
    OutputTooLarge,
    TokenAlignmentLoss,
    PartialProjection,
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarFallbackReason {
    ExternalUnconfigured,
    UnsupportedVariety,
    ExternalFailure,
    ProjectionIncomplete,
    ExternalRejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarBackendIdentity {
    pub backend: GrammarBackend,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
    #[serde(default)]
    pub configured_varieties: Vec<String>,
}

impl GrammarBackendIdentity {
    pub fn native() -> Self {
        Self {
            backend: GrammarBackend::TonguesRules,
            command: None,
            version: Some(env!("CARGO_PKG_VERSION").into()),
            model_path: None,
            model_sha256: None,
            configured_varieties: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarProjectionReport {
    pub input_tokens: usize,
    pub backend_tokens: usize,
    pub aligned_tokens: usize,
    #[serde(default)]
    pub unmatched_input_indices: Vec<usize>,
    #[serde(default)]
    pub unmatched_backend_tokens: Vec<BackendTokenIdentity>,
    #[serde(default)]
    pub dropped_backend_links: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarCoverageReport {
    pub input_tokens: usize,
    pub linked_tokens: usize,
    #[serde(default)]
    pub unsupported_token_indices: Vec<usize>,
    #[serde(default)]
    pub unsupported_constructs: Vec<String>,
}

impl GrammarProjectionReport {
    pub fn is_complete(&self) -> bool {
        self.input_tokens == self.aligned_tokens
            && self.backend_tokens == self.aligned_tokens
            && self.unmatched_input_indices.is_empty()
            && self.unmatched_backend_tokens.is_empty()
            && self.dropped_backend_links == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendTokenIdentity {
    pub id: usize,
    pub form: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarBackendReadiness {
    pub backend: GrammarBackend,
    pub state: GrammarBackendState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<GrammarBackendIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarBackendCatalog {
    pub variety: VarietyId,
    pub auto_policy: String,
    pub backends: Vec<GrammarBackendReadiness>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdPipeExecutionLimits {
    pub timeout: Duration,
    pub max_input_bytes: usize,
    pub max_stdout_bytes: usize,
    pub max_stderr_bytes: usize,
    pub poll_interval: Duration,
}

impl Default for UdPipeExecutionLimits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(DEFAULT_UDPIPE_TIMEOUT_MS),
            max_input_bytes: DEFAULT_UDPIPE_MAX_INPUT_BYTES,
            max_stdout_bytes: DEFAULT_UDPIPE_MAX_STDOUT_BYTES,
            max_stderr_bytes: DEFAULT_UDPIPE_MAX_STDERR_BYTES,
            poll_interval: Duration::from_millis(5),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdPipeBackendConfig {
    pub model_path: PathBuf,
    pub command: String,
    pub configured_varieties: Vec<String>,
    pub limits: UdPipeExecutionLimits,
}

impl UdPipeBackendConfig {
    pub fn identity(&self) -> GrammarBackendIdentity {
        GrammarBackendIdentity {
            backend: GrammarBackend::UdPipe,
            command: Some(self.command.clone()),
            version: command_version(&self.command, self.limits),
            model_path: Some(self.model_path.display().to_string()),
            model_sha256: model_sha256(&self.model_path),
            configured_varieties: self.configured_varieties.clone(),
        }
    }

    pub fn supports(&self, variety: &VarietyId) -> bool {
        self.configured_varieties
            .iter()
            .any(|configured| configured == &variety.0)
    }
}

pub(crate) struct UdPipeExecution {
    pub state: GrammarBackendState,
    pub stdout: Vec<u8>,
    pub stderr: String,
    pub status: Option<ExitStatus>,
    pub duration_ms: u64,
}

pub fn grammar_backend_catalog(variety: VarietyId) -> GrammarBackendCatalog {
    GrammarBackendCatalog {
        backends: vec![
            native_readiness(&variety),
            udpipe_readiness(&variety),
            link_grammar_readiness(&variety),
        ],
        variety,
        auto_policy: "use UDPipe only when configured for the requested variety and its projection is complete; otherwise retain the external attempt and fall back to native rules".into(),
    }
}

pub(crate) fn discover_udpipe_config(
    variety: &VarietyId,
) -> Result<UdPipeBackendConfig, Box<GrammarBackendReadiness>> {
    let scoped_name = scoped_model_environment_name(variety);
    let scoped_path = std::env::var(&scoped_name)
        .ok()
        .filter(|value| !value.trim().is_empty());
    let (model_path, configured_varieties) = if let Some(path) = scoped_path {
        (PathBuf::from(path), vec![variety.0.clone()])
    } else {
        let Some(path) = std::env::var("TONGUES_UDPIPE_MODEL")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            return Err(Box::new(GrammarBackendReadiness {
                backend: GrammarBackend::UdPipe,
                state: GrammarBackendState::UnavailableModel,
                diagnostic: Some(format!(
                    "set {scoped_name} for this variety or TONGUES_UDPIPE_MODEL with TONGUES_UDPIPE_MODEL_VARIETIES"
                )),
                identity: None,
            }));
        };
        let configured_varieties = std::env::var("TONGUES_UDPIPE_MODEL_VARIETIES")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        (PathBuf::from(path), configured_varieties)
    };
    let config = UdPipeBackendConfig {
        model_path,
        command: std::env::var("TONGUES_UDPIPE_COMMAND")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "udpipe".into()),
        configured_varieties,
        limits: UdPipeExecutionLimits::default(),
    };
    if !config.supports(variety) {
        return Err(Box::new(GrammarBackendReadiness {
            backend: GrammarBackend::UdPipe,
            state: GrammarBackendState::UnsupportedVariety,
            diagnostic: Some(format!(
                "configured UDPipe model does not declare support for {}",
                variety.0
            )),
            identity: Some(config.identity()),
        }));
    }
    if !config.model_path.is_file() {
        return Err(Box::new(GrammarBackendReadiness {
            backend: GrammarBackend::UdPipe,
            state: GrammarBackendState::UnavailableModel,
            diagnostic: Some(format!(
                "configured UDPipe model is not a readable file: {}",
                config.model_path.display()
            )),
            identity: Some(config.identity()),
        }));
    }
    Ok(config)
}

pub(crate) fn execute_udpipe(
    config: &UdPipeBackendConfig,
    input: &[u8],
    cancelled: Option<&AtomicBool>,
) -> UdPipeExecution {
    let args = vec![
        "--input=horizontal".into(),
        "--tag".into(),
        "--parse".into(),
        config.model_path.display().to_string(),
    ];
    execute_bounded_command(
        &config.command,
        &args,
        input,
        &[config.model_path.as_path()],
        config.limits,
        cancelled,
        "UDPipe",
    )
}

pub(crate) fn execute_bounded_command(
    command: &str,
    args: &[String],
    input: &[u8],
    redacted_paths: &[&Path],
    limits: UdPipeExecutionLimits,
    cancelled: Option<&AtomicBool>,
    backend_name: &str,
) -> UdPipeExecution {
    let started = Instant::now();
    if input.len() > limits.max_input_bytes {
        return execution_failure(
            GrammarBackendState::InputTooLarge,
            format!(
                "{backend_name} input was {} bytes; limit is {}",
                input.len(),
                limits.max_input_bytes
            ),
            started,
        );
    }
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return execution_failure(
            GrammarBackendState::Cancelled,
            format!("{backend_name} request was cancelled before spawn"),
            started,
        );
    }

    let mut child = match Command::new(command)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return execution_failure(
                GrammarBackendState::SpawnFailure,
                format!("failed to spawn {backend_name}: {error}"),
                started,
            );
        }
    };

    let stdout_reader = child
        .stdout
        .take()
        .map(|stdout| read_bounded(stdout, limits.max_stdout_bytes));
    let stderr_reader = child
        .stderr
        .take()
        .map(|stderr| read_bounded(stderr, limits.max_stderr_bytes));
    let stdin_writer = child.stdin.take().map(|mut stdin| {
        let input = input.to_vec();
        let backend_name = backend_name.to_string();
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let result = stdin
                .write_all(&input)
                .map_err(|error| format!("failed to write {backend_name} input: {error}"));
            let _ = sender.send(result);
        });
        receiver
    });

    let (state, status) = wait_bounded(&mut child, limits, cancelled);
    let stdin_error =
        stdin_writer.and_then(
            |writer| match writer.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error),
                Err(_) => Some(format!(
                    "{backend_name} stdin writer did not terminate after child exit"
                )),
            },
        );
    let stdout = join_reader(stdout_reader);
    let stderr = join_reader(stderr_reader);
    let mut redacted_stderr = redact_stderr(&stderr.bytes, redacted_paths);
    if stderr.truncated {
        redacted_stderr.push_str(" [stderr truncated]");
    }
    if state != GrammarBackendState::Accepted {
        return UdPipeExecution {
            state,
            stdout: stdout.bytes,
            stderr: redacted_stderr,
            status,
            duration_ms: elapsed_ms(started),
        };
    }
    if let Some(stdin_error) = stdin_error {
        if !redacted_stderr.is_empty() {
            redacted_stderr.push_str("; ");
        }
        redacted_stderr.push_str(&stdin_error);
        return UdPipeExecution {
            state: GrammarBackendState::Rejected,
            stdout: stdout.bytes,
            stderr: redacted_stderr,
            status,
            duration_ms: elapsed_ms(started),
        };
    }
    if stdout.truncated {
        return UdPipeExecution {
            state: GrammarBackendState::OutputTooLarge,
            stdout: stdout.bytes,
            stderr: redacted_stderr,
            status,
            duration_ms: elapsed_ms(started),
        };
    }
    if !status.is_some_and(|status| status.success()) {
        return UdPipeExecution {
            state: GrammarBackendState::Rejected,
            stdout: stdout.bytes,
            stderr: redacted_stderr,
            status,
            duration_ms: elapsed_ms(started),
        };
    }
    UdPipeExecution {
        state: GrammarBackendState::Accepted,
        stdout: stdout.bytes,
        stderr: redacted_stderr,
        status,
        duration_ms: elapsed_ms(started),
    }
}

fn native_readiness(variety: &VarietyId) -> GrammarBackendReadiness {
    match variety_by_code(&variety.0) {
        Some(variety) if variety.syntax_analyzer.is_some() || variety.syntax_rules.is_some() => {
            GrammarBackendReadiness {
                backend: GrammarBackend::TonguesRules,
                state: GrammarBackendState::Ready,
                diagnostic: None,
                identity: Some(GrammarBackendIdentity::native()),
            }
        }
        Some(_) => GrammarBackendReadiness {
            backend: GrammarBackend::TonguesRules,
            state: GrammarBackendState::UnsupportedVariety,
            diagnostic: Some(format!("{} declares no native grammar profile", variety.0)),
            identity: Some(GrammarBackendIdentity::native()),
        },
        None => GrammarBackendReadiness {
            backend: GrammarBackend::TonguesRules,
            state: GrammarBackendState::UnsupportedVariety,
            diagnostic: Some(format!("unknown linguistic variety {}", variety.0)),
            identity: Some(GrammarBackendIdentity::native()),
        },
    }
}

fn udpipe_readiness(variety: &VarietyId) -> GrammarBackendReadiness {
    match discover_udpipe_config(variety) {
        Ok(config) => {
            let identity = config.identity();
            let state = if identity.version.is_some() {
                GrammarBackendState::Ready
            } else {
                GrammarBackendState::SpawnFailure
            };
            GrammarBackendReadiness {
                backend: GrammarBackend::UdPipe,
                state,
                diagnostic: (state == GrammarBackendState::SpawnFailure)
                    .then(|| "UDPipe command did not return a bounded version probe".into()),
                identity: Some(identity),
            }
        }
        Err(readiness) => *readiness,
    }
}

fn scoped_model_environment_name(variety: &VarietyId) -> String {
    format!(
        "TONGUES_UDPIPE_MODEL_{}",
        variety
            .0
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character.to_ascii_uppercase()
                } else {
                    '_'
                }
            })
            .collect::<String>()
    )
}

fn wait_bounded(
    child: &mut Child,
    limits: UdPipeExecutionLimits,
    cancelled: Option<&AtomicBool>,
) -> (GrammarBackendState, Option<ExitStatus>) {
    let deadline = Instant::now() + limits.timeout;
    loop {
        if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            terminate_child(child);
            return (GrammarBackendState::Cancelled, child.wait().ok());
        }
        match child.try_wait() {
            Ok(Some(status)) => return (GrammarBackendState::Accepted, Some(status)),
            Ok(None) if Instant::now() >= deadline => {
                terminate_child(child);
                return (GrammarBackendState::Timeout, child.wait().ok());
            }
            Ok(None) => thread::sleep(limits.poll_interval),
            Err(_) => {
                terminate_child(child);
                return (GrammarBackendState::Rejected, child.wait().ok());
            }
        }
    }
}

fn terminate_child(child: &mut Child) {
    let _ = child.kill();
}

#[derive(Default)]
struct BoundedRead {
    bytes: Vec<u8>,
    truncated: bool,
}

fn read_bounded(mut reader: impl Read + Send + 'static, limit: usize) -> Receiver<BoundedRead> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let mut result = BoundedRead::default();
        let mut buffer = [0_u8; 8 * 1024];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(count) => {
                    let remaining = limit.saturating_sub(result.bytes.len());
                    result
                        .bytes
                        .extend_from_slice(&buffer[..count.min(remaining)]);
                    result.truncated |= count > remaining;
                }
            }
        }
        let _ = sender.send(result);
    });
    receiver
}

fn join_reader(reader: Option<Receiver<BoundedRead>>) -> BoundedRead {
    reader
        .and_then(|reader| reader.recv_timeout(Duration::from_millis(100)).ok())
        .unwrap_or_default()
}

fn redact_stderr(stderr: &[u8], paths: &[&Path]) -> String {
    let mut stderr = String::from_utf8_lossy(stderr).into_owned();
    for path in paths {
        stderr = stderr.replace(&path.display().to_string(), "[model]");
    }
    stderr
        .split_whitespace()
        .map(|token| {
            let lowercase = token.to_ascii_lowercase();
            if ["token=", "key=", "secret=", "password="]
                .iter()
                .any(|marker| lowercase.contains(marker))
            {
                "[redacted]"
            } else {
                token
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn execution_failure(
    state: GrammarBackendState,
    diagnostic: String,
    started: Instant,
) -> UdPipeExecution {
    UdPipeExecution {
        state,
        stdout: Vec::new(),
        stderr: diagnostic,
        status: None,
        duration_ms: elapsed_ms(started),
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

type ChecksumKey = (PathBuf, u64, Option<SystemTime>);
static MODEL_CHECKSUMS: OnceLock<Mutex<HashMap<ChecksumKey, String>>> = OnceLock::new();

pub(crate) fn model_sha256(path: &Path) -> Option<String> {
    let metadata = path.metadata().ok()?;
    let key = (path.to_path_buf(), metadata.len(), metadata.modified().ok());
    if let Some(checksum) = MODEL_CHECKSUMS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()?
        .get(&key)
        .cloned()
    {
        return Some(checksum);
    }
    let mut file = File::open(path).ok()?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).ok()?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let checksum = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    MODEL_CHECKSUMS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()?
        .insert(key, checksum.clone());
    Some(checksum)
}

static COMMAND_VERSIONS: OnceLock<Mutex<HashMap<String, Option<String>>>> = OnceLock::new();

pub(crate) fn command_version(command: &str, limits: UdPipeExecutionLimits) -> Option<String> {
    if let Some(version) = COMMAND_VERSIONS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()?
        .get(command)
        .cloned()
    {
        return version;
    }
    let probe_limits = UdPipeExecutionLimits {
        timeout: limits.timeout.min(Duration::from_millis(500)),
        max_input_bytes: 0,
        max_stdout_bytes: 4 * 1024,
        max_stderr_bytes: 4 * 1024,
        poll_interval: limits.poll_interval,
    };
    let version = bounded_version_probe(command, probe_limits);
    COMMAND_VERSIONS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()?
        .insert(command.into(), version.clone());
    version
}

fn bounded_version_probe(command: &str, limits: UdPipeExecutionLimits) -> Option<String> {
    let mut child = Command::new(command)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let stdout_reader = child
        .stdout
        .take()
        .map(|stdout| read_bounded(stdout, limits.max_stdout_bytes));
    let stderr_reader = child
        .stderr
        .take()
        .map(|stderr| read_bounded(stderr, limits.max_stderr_bytes));
    let (state, status) = wait_bounded(&mut child, limits, None);
    let stdout = join_reader(stdout_reader);
    let stderr = join_reader(stderr_reader);
    if state != GrammarBackendState::Accepted || !status.is_some_and(|status| status.success()) {
        return None;
    }
    let bytes = if stdout.bytes.is_empty() {
        stderr.bytes
    } else {
        stdout.bytes
    };
    let version = String::from_utf8_lossy(&bytes)
        .split_whitespace()
        .take(12)
        .collect::<Vec<_>>()
        .join(" ");
    (!version.is_empty()).then_some(version)
}
