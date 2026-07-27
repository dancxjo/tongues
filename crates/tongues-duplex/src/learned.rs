//! Trainable text-prefix continuation model and artifact lifecycle.
//!
//! The first learned vertical slice deliberately uses a compact, inspectable
//! categorical transducer. It learns continuation and policy distributions
//! from the versioned duplex rows, supports exact cached inference, and keeps
//! every speculative TTS segment held behind the existing evidence boundary.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use speaking::{
    CompletionHypothesisId, EvidenceProvenance, EvidenceSource, SentenceSyntaxAnalysis,
};
use tongues_data::check_group_split_leakage;

use crate::{
    CompletionMorpheme, CompletionProposal, CompletionProvider, CompletionProviderError,
    CompletionRequest, DuplexTrainingRow, TrainingRowKind, normalize_key, tokenize_morphemes,
};

pub const DUPLEX_MODEL_SCHEMA_VERSION: u32 = 1;
pub const DUPLEX_CHECKPOINT_FILE: &str = "checkpoint.json";
pub const DUPLEX_MANIFEST_FILE: &str = "manifest.json";
pub const DUPLEX_MODEL_CARD_FILE: &str = "MODEL_CARD.md";

fn default_epochs() -> u64 {
    1
}
fn default_max_continuation() -> usize {
    8
}
fn default_beam_width() -> usize {
    3
}
fn default_commit_threshold() -> f64 {
    0.72
}
fn default_withdraw_threshold() -> f64 {
    0.62
}
fn default_repair_threshold() -> f64 {
    0.68
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedDuplexConfig {
    #[serde(default = "default_epochs")]
    pub epochs: u64,
    #[serde(default = "default_max_continuation")]
    pub max_continuation_morphemes: usize,
    #[serde(default = "default_beam_width")]
    pub beam_width: usize,
    #[serde(default = "default_commit_threshold")]
    pub commit_threshold: f64,
    #[serde(default = "default_withdraw_threshold")]
    pub withdraw_threshold: f64,
    #[serde(default = "default_repair_threshold")]
    pub repair_threshold: f64,
    #[serde(default)]
    pub seed: u64,
}

impl Default for LearnedDuplexConfig {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            max_continuation_morphemes: default_max_continuation(),
            beam_width: default_beam_width(),
            commit_threshold: default_commit_threshold(),
            withdraw_threshold: default_withdraw_threshold(),
            repair_threshold: default_repair_threshold(),
            seed: 107,
        }
    }
}

impl LearnedDuplexConfig {
    fn validate(&self) -> Result<()> {
        ensure!(self.epochs > 0, "epochs must be greater than zero");
        ensure!(
            self.max_continuation_morphemes > 0,
            "max_continuation_morphemes must be greater than zero"
        );
        ensure!(self.beam_width > 0, "beam_width must be greater than zero");
        for (name, value) in [
            ("commit_threshold", self.commit_threshold),
            ("withdraw_threshold", self.withdraw_threshold),
            ("repair_threshold", self.repair_threshold),
        ] {
            ensure!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "{name} must be finite and within [0, 1]"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnedDecision {
    Commit,
    Hold,
    Withdraw,
    Repair,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct PrefixKey {
    committed: Vec<String>,
    unstable_suffix: Vec<String>,
}

impl PrefixKey {
    fn new(committed: &[String], unstable_suffix: &[String]) -> Self {
        Self {
            committed: normalize_tokens(committed),
            unstable_suffix: normalize_tokens(unstable_suffix),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
struct LearnedBucket {
    continuations: Vec<LearnedContinuationCount>,
    decision_counts: BTreeMap<LearnedDecision, u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct LearnedContinuationCount {
    morphemes: Vec<String>,
    count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct LearnedPrefixBucket {
    key: PrefixKey,
    bucket: LearnedBucket,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DuplexOptimizerState {
    pub observations: u64,
    pub momentum: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexSchedulerState {
    pub step: u64,
    pub learning_rate: f64,
    pub decay: f64,
}

impl Default for DuplexSchedulerState {
    fn default() -> Self {
        Self {
            step: 0,
            learning_rate: 1.0,
            decay: 0.98,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexDatasetProvenance {
    pub train_path: String,
    pub train_sha256: String,
    pub valid_path: Option<String>,
    pub valid_sha256: Option<String>,
    pub test_path: Option<String>,
    pub test_sha256: Option<String>,
    pub split_group_key: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedDuplexCheckpoint {
    pub schema_version: u32,
    pub family: String,
    pub architecture: String,
    pub epoch: u64,
    pub step: u64,
    pub config: LearnedDuplexConfig,
    pub vocabulary: Vec<String>,
    buckets: Vec<LearnedPrefixBucket>,
    global: LearnedBucket,
    pub optimizer: DuplexOptimizerState,
    pub scheduler: DuplexSchedulerState,
    pub provenance: DuplexDatasetProvenance,
    pub latest_metrics: Option<DuplexEvaluationReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContinuationCandidate {
    pub morphemes: Vec<String>,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedInference {
    pub candidates: Vec<ContinuationCandidate>,
    pub decision: LearnedDecision,
    pub decision_confidence: f64,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexCalibrationMetrics {
    pub brier_score: f64,
    pub expected_calibration_error: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexBehaviorMetrics {
    pub decision_accuracy: f64,
    pub predicted_commit: usize,
    pub predicted_hold: usize,
    pub predicted_withdraw: usize,
    pub predicted_repair: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexEvaluationReport {
    pub rows: usize,
    pub groups: usize,
    pub continuation_token_f1: f64,
    pub exact_continuation_match: f64,
    pub deterministic_baseline_token_f1: f64,
    pub named_held_out_metric: String,
    pub named_metric_improved: bool,
    pub calibration: DuplexCalibrationMetrics,
    pub behavior: DuplexBehaviorMetrics,
    pub latency_mean_micros: f64,
    pub latency_p95_micros: u128,
    pub cached_uncached_equivalent: bool,
    pub playback_safety_violations: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexTrainReport {
    pub checkpoint: String,
    pub epoch: u64,
    pub step: u64,
    pub train_rows: usize,
    pub vocabulary_size: usize,
    pub metrics: Option<DuplexEvaluationReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplexArtifactManifest {
    pub schema_version: u32,
    pub family: String,
    pub architecture: String,
    pub checkpoint: String,
    pub model_card: String,
    pub epoch: u64,
    pub vocabulary_size: usize,
    pub provenance: DuplexDatasetProvenance,
    pub metrics: Option<DuplexEvaluationReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredDuplexModel {
    pub artifact_dir: String,
    pub manifest: DuplexArtifactManifest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsPlanDisposition {
    Held,
    Playable,
    Withdraw,
    Repair,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsPlanDelta {
    pub hypothesis_id: String,
    pub disposition: TtsPlanDisposition,
    pub text: String,
    pub evidence_ids: Vec<String>,
}

/// Converts inference into TTS deltas without bypassing the playback boundary.
///
/// Predicted text is always `Held`. Only text whose morphemes all carry direct
/// observation IDs may be `Playable`.
#[derive(Debug, Clone, Default)]
pub struct TtsPlanDeltaPredictor;

impl TtsPlanDeltaPredictor {
    pub fn predicted(
        &self,
        hypothesis_id: impl Into<String>,
        text: impl Into<String>,
    ) -> TtsPlanDelta {
        TtsPlanDelta {
            hypothesis_id: hypothesis_id.into(),
            disposition: TtsPlanDisposition::Held,
            text: text.into(),
            evidence_ids: Vec::new(),
        }
    }

    pub fn observed(
        &self,
        hypothesis_id: impl Into<String>,
        morphemes: &[CompletionMorpheme],
    ) -> TtsPlanDelta {
        let evidence_ids = morphemes
            .iter()
            .flat_map(|morpheme| morpheme.evidence.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let all_observed = !morphemes.is_empty()
            && morphemes
                .iter()
                .all(|morpheme| !morpheme.evidence.is_empty());
        TtsPlanDelta {
            hypothesis_id: hypothesis_id.into(),
            disposition: if all_observed {
                TtsPlanDisposition::Playable
            } else {
                TtsPlanDisposition::Held
            },
            text: morphemes
                .iter()
                .map(|morpheme| morpheme.surface.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            evidence_ids,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LearnedDuplexModel {
    checkpoint: LearnedDuplexCheckpoint,
    cache: BTreeMap<PrefixKey, LearnedInference>,
}

impl LearnedDuplexModel {
    pub fn load(path: &Path) -> Result<Self> {
        let path = checkpoint_path(path);
        let checkpoint: LearnedDuplexCheckpoint = read_json(&path)?;
        ensure!(
            checkpoint.schema_version == DUPLEX_MODEL_SCHEMA_VERSION,
            "incompatible duplex checkpoint schema {} at {}; expected {}",
            checkpoint.schema_version,
            path.display(),
            DUPLEX_MODEL_SCHEMA_VERSION
        );
        checkpoint.config.validate()?;
        Ok(Self {
            checkpoint,
            cache: BTreeMap::new(),
        })
    }

    pub fn checkpoint(&self) -> &LearnedDuplexCheckpoint {
        &self.checkpoint
    }

    pub fn infer_uncached(
        &self,
        committed: &[String],
        unstable_suffix: &[String],
    ) -> LearnedInference {
        infer_checkpoint(&self.checkpoint, committed, unstable_suffix, false)
    }

    pub fn infer_cached(
        &mut self,
        committed: &[String],
        unstable_suffix: &[String],
    ) -> LearnedInference {
        let key = PrefixKey::new(committed, unstable_suffix);
        if let Some(cached) = self.cache.get(&key) {
            let mut cached = cached.clone();
            cached.cache_hit = true;
            return cached;
        }
        let inference = infer_checkpoint(&self.checkpoint, committed, unstable_suffix, false);
        self.cache.insert(key, inference.clone());
        inference
    }
}

#[derive(Debug, Clone)]
pub struct LearnedCompletionProvider {
    model: LearnedDuplexModel,
    unstable_by_utterance: BTreeMap<String, Vec<String>>,
}

impl LearnedCompletionProvider {
    pub fn from_checkpoint(path: &Path) -> Result<Self> {
        Ok(Self {
            model: LearnedDuplexModel::load(path)?,
            unstable_by_utterance: BTreeMap::new(),
        })
    }

    pub fn from_artifact(path: &Path) -> Result<Self> {
        let manifest_path = if path.is_dir() {
            path.join(DUPLEX_MANIFEST_FILE)
        } else {
            path.to_path_buf()
        };
        let manifest: DuplexArtifactManifest = read_json(&manifest_path)?;
        ensure!(
            manifest.family == "duplex",
            "artifact is not a duplex model"
        );
        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        Self::from_checkpoint(&root.join(manifest.checkpoint))
    }
}

impl CompletionProvider for LearnedCompletionProvider {
    fn complete(
        &mut self,
        request: &CompletionRequest,
    ) -> std::result::Result<Vec<CompletionProposal>, CompletionProviderError> {
        let observed = observed_morphemes(request);
        let committed = request
            .committed
            .iter()
            .map(|morpheme| morpheme.surface.clone())
            .collect::<Vec<_>>();
        let utterance_key = request.utterance_id.0.clone();
        let unstable = self
            .unstable_by_utterance
            .get(&utterance_key)
            .cloned()
            .unwrap_or_default();
        let inference = self.model.infer_cached(&committed, &unstable);
        let next_unstable = inference
            .candidates
            .first()
            .map(|candidate| candidate.morphemes.clone())
            .unwrap_or_default();
        self.unstable_by_utterance
            .insert(utterance_key, next_unstable);

        let provenance = EvidenceProvenance {
            source: EvidenceSource::Inference,
            method: "learned-duplex-prefix-transducer".into(),
            version: Some(self.model.checkpoint.epoch.to_string()),
        };
        let evidence = request
            .evidence
            .iter()
            .map(|evidence| evidence.id.clone())
            .collect::<Vec<_>>();
        let candidates = if inference.candidates.is_empty() {
            vec![ContinuationCandidate {
                morphemes: Vec::new(),
                probability: 1.0,
            }]
        } else {
            inference.candidates
        };
        Ok(candidates
            .into_iter()
            .enumerate()
            .map(|(index, candidate)| {
                let mut morphemes = observed.clone();
                morphemes.extend(candidate.morphemes.into_iter().map(|surface| {
                    CompletionMorpheme::predicted(
                        normalize_key(&surface),
                        surface,
                        request.variety.clone(),
                    )
                }));
                CompletionProposal {
                    id: CompletionHypothesisId(format!("learned:{index}")),
                    weight: candidate.probability.max(f64::EPSILON),
                    morphemes,
                    syntax: Some(SentenceSyntaxAnalysis::default()),
                    prosody: None,
                    evidence: evidence.clone(),
                    provenance: provenance.clone(),
                }
            })
            .collect())
    }
}

pub fn train_duplex_model(
    data_dir: &Path,
    run_dir: &Path,
    config: LearnedDuplexConfig,
    resume: bool,
    mut progress: impl FnMut(String),
) -> Result<DuplexTrainReport> {
    config.validate()?;
    let train_path = data_dir.join("train.jsonl");
    let valid_path = optional_path(&data_dir.join("valid.jsonl"));
    let test_path = optional_path(&data_dir.join("test.jsonl"));
    let train = read_rows(&train_path)?;
    ensure!(!train.is_empty(), "training split is empty");
    let valid = valid_path
        .as_deref()
        .map(read_rows)
        .transpose()?
        .unwrap_or_default();
    let test = test_path
        .as_deref()
        .map(read_rows)
        .transpose()?
        .unwrap_or_default();
    validate_split_groups(&train, &valid, &test)?;

    fs::create_dir_all(run_dir).with_context(|| format!("creating {}", run_dir.display()))?;
    let checkpoint_file = run_dir.join(DUPLEX_CHECKPOINT_FILE);
    let provenance = DuplexDatasetProvenance {
        train_path: train_path.display().to_string(),
        train_sha256: sha256_file(&train_path)?,
        valid_path: valid_path.as_ref().map(|path| path.display().to_string()),
        valid_sha256: valid_path.as_deref().map(sha256_file).transpose()?,
        test_path: test_path.as_ref().map(|path| path.display().to_string()),
        test_sha256: test_path.as_deref().map(sha256_file).transpose()?,
        split_group_key: "fixture_id".into(),
    };
    let mut checkpoint = if resume {
        ensure!(
            checkpoint_file.exists(),
            "cannot resume: {} does not exist",
            checkpoint_file.display()
        );
        let mut checkpoint: LearnedDuplexCheckpoint = read_json(&checkpoint_file)?;
        ensure!(
            checkpoint.schema_version == DUPLEX_MODEL_SCHEMA_VERSION,
            "cannot resume incompatible checkpoint schema {}",
            checkpoint.schema_version
        );
        ensure!(
            checkpoint.provenance.train_sha256 == provenance.train_sha256,
            "cannot resume with different training data (checkpoint {}, current {})",
            checkpoint.provenance.train_sha256,
            provenance.train_sha256
        );
        checkpoint.config = config.clone();
        checkpoint.provenance = provenance;
        checkpoint
    } else {
        LearnedDuplexCheckpoint {
            schema_version: DUPLEX_MODEL_SCHEMA_VERSION,
            family: "duplex".into(),
            architecture: "categorical-text-prefix-transducer-v1".into(),
            epoch: 0,
            step: 0,
            config: config.clone(),
            vocabulary: Vec::new(),
            buckets: Vec::new(),
            global: LearnedBucket::default(),
            optimizer: DuplexOptimizerState::default(),
            scheduler: DuplexSchedulerState::default(),
            provenance,
            latest_metrics: None,
        }
    };

    progress(format!(
        "checkpoint={} epoch_pattern=model-epoch-N.json best_model={}",
        checkpoint_file.display(),
        run_dir.join("model.json").display()
    ));
    let mut vocabulary = checkpoint
        .vocabulary
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    for _ in 0..config.epochs {
        let next_epoch = checkpoint.epoch + 1;
        progress(format!(
            "training epoch {next_epoch} over {} rows -> {}",
            train.len(),
            run_dir.display()
        ));
        for (index, row) in train.iter().enumerate() {
            update_checkpoint(&mut checkpoint, row, &mut vocabulary);
            if index < 3 || (index + 1) % 100 == 0 {
                progress(format!(
                    "epoch {next_epoch}: learned {}/{} rows -> {}",
                    index + 1,
                    train.len(),
                    checkpoint_file.display()
                ));
            }
        }
        checkpoint.epoch = next_epoch;
        checkpoint.scheduler.step = checkpoint.scheduler.step.saturating_add(1);
        checkpoint.scheduler.learning_rate *= checkpoint.scheduler.decay;
        checkpoint.vocabulary = vocabulary.iter().cloned().collect();
        let epoch_file = run_dir.join(format!("model-epoch-{}.json", checkpoint.epoch));
        write_json_atomic(&epoch_file, &checkpoint)?;
        write_json_atomic(&checkpoint_file, &checkpoint)?;
    }

    let eval_rows = if !valid.is_empty() { &valid } else { &test };
    let metrics = if eval_rows.is_empty() {
        None
    } else {
        Some(evaluate_checkpoint(&checkpoint, eval_rows)?)
    };
    checkpoint.latest_metrics = metrics.clone();
    write_json_atomic(&run_dir.join("model.json"), &checkpoint)?;
    write_json_atomic(&checkpoint_file, &checkpoint)?;
    write_json_atomic(
        &run_dir.join("train_state.json"),
        &serde_json::json!({
            "epoch": checkpoint.epoch,
            "step": checkpoint.step,
            "optimizer": checkpoint.optimizer,
            "scheduler": checkpoint.scheduler,
            "latest_metrics": checkpoint.latest_metrics,
        }),
    )?;
    Ok(DuplexTrainReport {
        checkpoint: checkpoint_file.display().to_string(),
        epoch: checkpoint.epoch,
        step: checkpoint.step,
        train_rows: train.len(),
        vocabulary_size: checkpoint.vocabulary.len(),
        metrics,
    })
}

pub fn evaluate_duplex_model(
    model_path: &Path,
    split_path: &Path,
) -> Result<DuplexEvaluationReport> {
    let model = LearnedDuplexModel::load(model_path)?;
    let rows = read_rows(split_path)?;
    ensure!(!rows.is_empty(), "evaluation split is empty");
    evaluate_checkpoint(&model.checkpoint, &rows)
}

pub fn export_duplex_model(model_path: &Path, out: &Path) -> Result<DuplexArtifactManifest> {
    let model = LearnedDuplexModel::load(model_path)?;
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    write_json_atomic(&out.join("model.json"), &model.checkpoint)?;
    let manifest = DuplexArtifactManifest {
        schema_version: DUPLEX_MODEL_SCHEMA_VERSION,
        family: "duplex".into(),
        architecture: model.checkpoint.architecture.clone(),
        checkpoint: "model.json".into(),
        model_card: DUPLEX_MODEL_CARD_FILE.into(),
        epoch: model.checkpoint.epoch,
        vocabulary_size: model.checkpoint.vocabulary.len(),
        provenance: model.checkpoint.provenance.clone(),
        metrics: model.checkpoint.latest_metrics.clone(),
    };
    write_json_atomic(&out.join(DUPLEX_MANIFEST_FILE), &manifest)?;
    let metric = manifest
        .metrics
        .as_ref()
        .map(|metrics| {
            format!(
                "- Held-out continuation token F1: {:.4}\n- Deterministic baseline token F1: {:.4}\n- Playback safety violations: {}\n",
                metrics.continuation_token_f1,
                metrics.deterministic_baseline_token_f1,
                metrics.playback_safety_violations
            )
        })
        .unwrap_or_else(|| "- No held-out metrics recorded.\n".into());
    let card = format!(
        "# Learned predictive duplex transducer\n\n\
         - Architecture: `{}`\n\
         - Epoch: {}\n\
         - Vocabulary: {} tokens\n\
         - Training SHA-256: `{}`\n\
         - Split group key: `{}`\n\n\
         ## Evaluation\n\n{}\
         ## Safety boundary\n\n\
         Predicted, unobserved continuations are emitted only as held TTS plan \
         deltas. Playback eligibility remains evidence-backed and is enforced by \
         the duplex simulator/runtime boundary.\n",
        manifest.architecture,
        manifest.epoch,
        manifest.vocabulary_size,
        manifest.provenance.train_sha256,
        manifest.provenance.split_group_key,
        metric,
    );
    write_text_atomic(&out.join(DUPLEX_MODEL_CARD_FILE), &card)?;
    Ok(manifest)
}

pub fn discover_duplex_models(root: &Path) -> Result<Vec<DiscoveredDuplexModel>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut manifests = Vec::new();
    discover_manifests(root, root, &mut manifests)?;
    manifests.sort_by(|left, right| left.artifact_dir.cmp(&right.artifact_dir));
    Ok(manifests)
}

fn discover_manifests(
    root: &Path,
    path: &Path,
    found: &mut Vec<DiscoveredDuplexModel>,
) -> Result<()> {
    if path.join(DUPLEX_MANIFEST_FILE).is_file() {
        let manifest: DuplexArtifactManifest = read_json(&path.join(DUPLEX_MANIFEST_FILE))?;
        if manifest.family == "duplex" {
            found.push(DiscoveredDuplexModel {
                artifact_dir: path
                    .strip_prefix(root)
                    .unwrap_or(path)
                    .display()
                    .to_string(),
                manifest,
            });
        }
        return Ok(());
    }
    for entry in fs::read_dir(path).with_context(|| format!("reading {}", path.display()))? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            discover_manifests(root, &entry.path(), found)?;
        }
    }
    Ok(())
}

fn evaluate_checkpoint(
    checkpoint: &LearnedDuplexCheckpoint,
    rows: &[DuplexTrainingRow],
) -> Result<DuplexEvaluationReport> {
    let mut token_f1 = 0.0;
    let mut baseline_f1 = 0.0;
    let mut exact = 0usize;
    let mut correct_decisions = 0usize;
    let mut brier = 0.0;
    let mut calibration_bins = vec![(0usize, 0.0f64, 0.0f64); 10];
    let mut behavior = DuplexBehaviorMetrics {
        decision_accuracy: 0.0,
        predicted_commit: 0,
        predicted_hold: 0,
        predicted_withdraw: 0,
        predicted_repair: 0,
    };
    let mut latencies = Vec::with_capacity(rows.len());
    let mut cache_model = LearnedDuplexModel {
        checkpoint: checkpoint.clone(),
        cache: BTreeMap::new(),
    };
    let mut equivalent = true;
    let planner = TtsPlanDeltaPredictor;
    let mut safety_violations = 0usize;

    for row in rows {
        let target = target_continuation(row);
        let start = Instant::now();
        let inference = infer_checkpoint(
            checkpoint,
            &row.committed_prefix,
            &row.predicted_suffix,
            false,
        );
        latencies.push(start.elapsed().as_micros());
        let predicted = inference
            .candidates
            .first()
            .map(|candidate| candidate.morphemes.as_slice())
            .unwrap_or(&[]);
        token_f1 += sequence_token_f1(predicted, &target);
        baseline_f1 += sequence_token_f1(&["<statement>".into()], &target);
        exact += usize::from(predicted == target.as_slice());

        let target_decision = target_decision(row);
        let correct = inference.decision == target_decision;
        correct_decisions += usize::from(correct);
        brier += (inference.decision_confidence - f64::from(correct)).powi(2);
        let bin = ((inference.decision_confidence * 10.0).floor() as usize).min(9);
        calibration_bins[bin].0 += 1;
        calibration_bins[bin].1 += inference.decision_confidence;
        calibration_bins[bin].2 += f64::from(correct);
        match inference.decision {
            LearnedDecision::Commit => behavior.predicted_commit += 1,
            LearnedDecision::Hold => behavior.predicted_hold += 1,
            LearnedDecision::Withdraw => behavior.predicted_withdraw += 1,
            LearnedDecision::Repair => behavior.predicted_repair += 1,
        }

        let uncached = cache_model.infer_uncached(&row.committed_prefix, &row.predicted_suffix);
        let first_cached = cache_model.infer_cached(&row.committed_prefix, &row.predicted_suffix);
        let second_cached = cache_model.infer_cached(&row.committed_prefix, &row.predicted_suffix);
        equivalent &= uncached.candidates == first_cached.candidates
            && uncached.decision == first_cached.decision
            && second_cached.cache_hit
            && uncached.candidates == second_cached.candidates;

        let plan = planner.predicted("evaluation", predicted.join(" "));
        safety_violations += usize::from(plan.disposition == TtsPlanDisposition::Playable);
    }
    let n = rows.len() as f64;
    latencies.sort_unstable();
    let p95_index = ((latencies.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(latencies.len().saturating_sub(1));
    let ece = calibration_bins
        .into_iter()
        .filter(|(count, _, _)| *count > 0)
        .map(|(count, confidence, accuracy)| {
            let count_f = count as f64;
            (count_f / n) * ((confidence / count_f) - (accuracy / count_f)).abs()
        })
        .sum();
    behavior.decision_accuracy = correct_decisions as f64 / n;
    let continuation_token_f1 = token_f1 / n;
    let deterministic_baseline_token_f1 = baseline_f1 / n;
    Ok(DuplexEvaluationReport {
        rows: rows.len(),
        groups: rows
            .iter()
            .map(|row| row.fixture_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        continuation_token_f1,
        exact_continuation_match: exact as f64 / n,
        deterministic_baseline_token_f1,
        named_held_out_metric: "continuation_token_f1".into(),
        named_metric_improved: continuation_token_f1 > deterministic_baseline_token_f1,
        calibration: DuplexCalibrationMetrics {
            brier_score: brier / n,
            expected_calibration_error: ece,
        },
        behavior,
        latency_mean_micros: latencies.iter().sum::<u128>() as f64 / n,
        latency_p95_micros: latencies[p95_index],
        cached_uncached_equivalent: equivalent,
        playback_safety_violations: safety_violations,
    })
}

fn update_checkpoint(
    checkpoint: &mut LearnedDuplexCheckpoint,
    row: &DuplexTrainingRow,
    vocabulary: &mut BTreeSet<String>,
) {
    let key = PrefixKey::new(&row.committed_prefix, &row.predicted_suffix);
    let continuation = target_continuation(row);
    let decision = target_decision(row);
    for token in row
        .committed_prefix
        .iter()
        .chain(row.predicted_suffix.iter())
        .chain(continuation.iter())
    {
        vocabulary.insert(normalize_key(token));
    }
    let bucket = if let Some(index) = checkpoint
        .buckets
        .iter()
        .position(|candidate| candidate.key == key)
    {
        &mut checkpoint.buckets[index].bucket
    } else {
        checkpoint.buckets.push(LearnedPrefixBucket {
            key,
            bucket: LearnedBucket::default(),
        });
        &mut checkpoint.buckets.last_mut().expect("just inserted").bucket
    };
    update_bucket(bucket, &continuation, decision);
    update_bucket(&mut checkpoint.global, &continuation, decision);
    checkpoint.step = checkpoint.step.saturating_add(1);
    checkpoint.optimizer.observations = checkpoint.optimizer.observations.saturating_add(1);
    *checkpoint
        .optimizer
        .momentum
        .entry(format!("{decision:?}").to_lowercase())
        .or_default() = 0.9
        * checkpoint
            .optimizer
            .momentum
            .get(&format!("{decision:?}").to_lowercase())
            .copied()
            .unwrap_or_default()
        + 0.1;
}

fn update_bucket(bucket: &mut LearnedBucket, continuation: &[String], decision: LearnedDecision) {
    let continuation = normalize_tokens(continuation);
    if let Some(existing) = bucket
        .continuations
        .iter_mut()
        .find(|candidate| candidate.morphemes == continuation)
    {
        existing.count += 1;
    } else {
        bucket.continuations.push(LearnedContinuationCount {
            morphemes: continuation,
            count: 1,
        });
    }
    *bucket.decision_counts.entry(decision).or_default() += 1;
}

fn infer_checkpoint(
    checkpoint: &LearnedDuplexCheckpoint,
    committed: &[String],
    unstable_suffix: &[String],
    cache_hit: bool,
) -> LearnedInference {
    let key = PrefixKey::new(committed, unstable_suffix);
    let bucket = checkpoint
        .buckets
        .iter()
        .find(|candidate| candidate.key == key)
        .map(|candidate| &candidate.bucket)
        .unwrap_or(&checkpoint.global);
    let mut continuations = bucket
        .continuations
        .iter()
        .map(|candidate| (candidate.morphemes.clone(), candidate.count))
        .collect::<Vec<_>>();
    continuations.sort_by(|(left_tokens, left_count), (right_tokens, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_tokens.cmp(right_tokens))
    });
    let total = continuations
        .iter()
        .map(|(_, count)| *count)
        .sum::<u64>()
        .max(1) as f64;
    let candidates = continuations
        .into_iter()
        .take(checkpoint.config.beam_width)
        .map(|(mut morphemes, count)| {
            morphemes.truncate(checkpoint.config.max_continuation_morphemes);
            ContinuationCandidate {
                morphemes,
                probability: count as f64 / total,
            }
        })
        .collect();
    let (raw_decision, confidence) = bucket
        .decision_counts
        .iter()
        .max_by(
            |(left_decision, left_count), (right_decision, right_count)| {
                left_count
                    .cmp(right_count)
                    .then_with(|| right_decision.cmp(left_decision))
            },
        )
        .map(|(decision, count)| {
            let total = bucket.decision_counts.values().sum::<u64>().max(1);
            (*decision, *count as f64 / total as f64)
        })
        .unwrap_or((LearnedDecision::Hold, 1.0));
    let threshold = match raw_decision {
        LearnedDecision::Commit => checkpoint.config.commit_threshold,
        LearnedDecision::Withdraw => checkpoint.config.withdraw_threshold,
        LearnedDecision::Repair => checkpoint.config.repair_threshold,
        LearnedDecision::Hold => 0.0,
    };
    LearnedInference {
        candidates,
        decision: if confidence >= threshold {
            raw_decision
        } else {
            LearnedDecision::Hold
        },
        decision_confidence: confidence,
        cache_hit,
    }
}

fn target_decision(row: &DuplexTrainingRow) -> LearnedDecision {
    match row.row_kind {
        TrainingRowKind::Rollback => LearnedDecision::Withdraw,
        TrainingRowKind::Repair => LearnedDecision::Repair,
        TrainingRowKind::Completion if row.commit_frontier < row.safe_commit_count => {
            LearnedDecision::Commit
        }
        _ if row.commit_frontier < row.safe_commit_count => LearnedDecision::Commit,
        _ => LearnedDecision::Hold,
    }
}

fn target_continuation(row: &DuplexTrainingRow) -> Vec<String> {
    let final_tokens = tokenize_morphemes(&row.final_committed_text);
    final_tokens.into_iter().skip(row.commit_frontier).collect()
}

fn observed_morphemes(request: &CompletionRequest) -> Vec<CompletionMorpheme> {
    let mut observed = Vec::new();
    for evidence in &request.evidence {
        let tokens = if evidence.supports.is_empty() {
            tokenize_morphemes(&evidence.content)
        } else {
            evidence.supports.clone()
        };
        for surface in tokens {
            observed.push(CompletionMorpheme {
                key: normalize_key(&surface),
                surface,
                variety: request.variety.clone(),
                evidence: vec![evidence.id.clone()],
            });
        }
    }
    observed
}

fn validate_split_groups(
    train: &[DuplexTrainingRow],
    valid: &[DuplexTrainingRow],
    test: &[DuplexTrainingRow],
) -> Result<()> {
    let groups = |rows: &[DuplexTrainingRow]| {
        rows.iter()
            .map(|row| row.fixture_id.clone())
            .collect::<Vec<_>>()
    };
    let conflicts = check_group_split_leakage(&[
        ("train", groups(train)),
        ("valid", groups(valid)),
        ("test", groups(test)),
    ]);
    if !conflicts.is_empty() {
        bail!(
            "duplex split leakage detected for group key fixture_id:\n{}",
            conflicts.join("\n")
        );
    }
    Ok(())
}

fn normalize_tokens(tokens: &[String]) -> Vec<String> {
    tokens.iter().map(|token| normalize_key(token)).collect()
}

fn sequence_token_f1(predicted: &[String], target: &[String]) -> f64 {
    if predicted.is_empty() && target.is_empty() {
        return 1.0;
    }
    if predicted.is_empty() || target.is_empty() {
        return 0.0;
    }
    let mut target_counts = BTreeMap::<String, usize>::new();
    for token in target {
        *target_counts.entry(normalize_key(token)).or_default() += 1;
    }
    let mut matches = 0usize;
    for token in predicted {
        if let Some(count) = target_counts.get_mut(&normalize_key(token))
            && *count > 0
        {
            matches += 1;
            *count -= 1;
        }
    }
    let precision = matches as f64 / predicted.len() as f64;
    let recall = matches as f64 / target.len() as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn read_rows(path: &Path) -> Result<Vec<DuplexTrainingRow>> {
    let reader =
        BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?);
    reader
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let line =
                line.with_context(|| format!("reading {} line {}", path.display(), index + 1))?;
            serde_json::from_str(&line)
                .with_context(|| format!("parsing {} line {}", path.display(), index + 1))
        })
        .collect()
}

fn optional_path(path: &Path) -> Option<PathBuf> {
    path.is_file().then(|| path.to_path_buf())
}

fn checkpoint_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        let preferred = path.join("model.json");
        if preferred.is_file() {
            preferred
        } else {
            path.join(DUPLEX_CHECKPOINT_FILE)
        }
    } else {
        path.to_path_buf()
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let part = part_path(path);
    let file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .with_context(|| format!("writing {}", part.display()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&part, path)
        .with_context(|| format!("renaming {} to {}", part.display(), path.display()))
}

fn write_text_atomic(path: &Path, value: &str) -> Result<()> {
    let part = part_path(path);
    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    file.write_all(value.as_bytes())?;
    file.sync_all()?;
    fs::rename(&part, path)
        .with_context(|| format!("renaming {} to {}", part.display(), path.display()))
}

fn part_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "duplex-artifact".into());
    name.push(".part");
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DuplexSimulator, ObservedEvidence, SimulatorConfig};
    use speaking::{UtteranceId, VarietyId};
    use tempfile::tempdir;

    fn prepared_data() -> (tempfile::TempDir, PathBuf) {
        let root = tempdir().unwrap();
        let data = root.path().join("data");
        crate::prepare_dataset(
            &data,
            Path::new("../../fixtures/duplex/completion_scenarios_v1.json"),
        )
        .unwrap();
        (root, data)
    }

    #[test]
    fn cached_inference_exactly_matches_uncached() {
        let (root, data) = prepared_data();
        let run = root.path().join("run");
        train_duplex_model(&data, &run, LearnedDuplexConfig::default(), false, |_| {}).unwrap();
        let mut model = LearnedDuplexModel::load(&run).unwrap();
        let committed = vec!["Who".into(), "shot".into()];
        let unstable = vec!["John".into()];
        let uncached = model.infer_uncached(&committed, &unstable);
        let first = model.infer_cached(&committed, &unstable);
        let second = model.infer_cached(&committed, &unstable);
        assert_eq!(uncached.candidates, first.candidates);
        assert_eq!(uncached.decision, first.decision);
        assert!(second.cache_hit);
        assert_eq!(uncached.candidates, second.candidates);
    }

    #[test]
    fn checkpoint_resume_restores_all_training_state() {
        let (root, data) = prepared_data();
        let run = root.path().join("run");
        let first =
            train_duplex_model(&data, &run, LearnedDuplexConfig::default(), false, |_| {}).unwrap();
        let before = LearnedDuplexModel::load(&run).unwrap().checkpoint().clone();
        let second =
            train_duplex_model(&data, &run, LearnedDuplexConfig::default(), true, |_| {}).unwrap();
        let after = LearnedDuplexModel::load(&run).unwrap().checkpoint().clone();
        assert_eq!(second.epoch, first.epoch + 1);
        assert!(after.step > before.step);
        assert!(after.optimizer.observations > before.optimizer.observations);
        assert!(after.scheduler.step > before.scheduler.step);
        assert_eq!(after.vocabulary, before.vocabulary);
        assert_eq!(after.config, before.config);
    }

    #[test]
    fn learned_provider_preserves_evidence_backed_commit_boundary() {
        let (root, data) = prepared_data();
        let run = root.path().join("run");
        train_duplex_model(&data, &run, LearnedDuplexConfig::default(), false, |_| {}).unwrap();
        let provider = LearnedCompletionProvider::from_checkpoint(&run).unwrap();
        let mut simulator = DuplexSimulator::new(
            UtteranceId("learned-test".into()),
            VarietyId("en-US".into()),
            SimulatorConfig::default(),
            provider,
        )
        .unwrap();
        simulator
            .observe(ObservedEvidence::text("e1", "Who shot"))
            .unwrap();
        assert!(
            simulator
                .state()
                .committed
                .iter()
                .all(|m| !m.evidence.is_empty())
        );
        assert!(
            simulator
                .state()
                .predicted_suffix()
                .iter()
                .all(|m| m.evidence.is_empty())
        );
    }

    #[test]
    fn export_is_discoverable_and_keeps_predicted_audio_held() {
        let (root, data) = prepared_data();
        let run = root.path().join("run");
        let out = root.path().join("models/duplex/example");
        train_duplex_model(&data, &run, LearnedDuplexConfig::default(), false, |_| {}).unwrap();
        export_duplex_model(&run, &out).unwrap();
        let discovered = discover_duplex_models(&root.path().join("models")).unwrap();
        assert_eq!(discovered.len(), 1);
        LearnedCompletionProvider::from_artifact(&out).unwrap();
        let delta = TtsPlanDeltaPredictor.predicted("h1", "unobserved continuation");
        assert_eq!(delta.disposition, TtsPlanDisposition::Held);
        assert!(delta.evidence_ids.is_empty());
    }
}
