//! Anonymous speaker diarization and explicit voice-familiarity contracts.

use serde::{Deserialize, Serialize};

use crate::{Confidence, ConfidenceScale, SegmentId, StreamEvent};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpeakerClusterId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VoiceSignatureId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceRetentionScope {
    Segment,
    Session,
    PersistentOptIn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceRetentionPolicy {
    pub retain_for_session: bool,
    pub retain_persistently: bool,
}

impl Default for VoiceRetentionPolicy {
    fn default() -> Self {
        Self {
            retain_for_session: true,
            retain_persistently: false,
        }
    }
}

impl VoiceRetentionPolicy {
    pub fn permits(self, scope: VoiceRetentionScope) -> bool {
        match scope {
            VoiceRetentionScope::Segment => true,
            VoiceRetentionScope::Session => self.retain_for_session,
            VoiceRetentionScope::PersistentOptIn => self.retain_persistently,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    pub values: Vec<f32>,
    pub model_id: String,
    pub retention: VoiceRetentionScope,
}

impl SpeakerEmbedding {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.values.is_empty(), "speaker embedding is empty");
        anyhow::ensure!(
            self.values.iter().all(|value| value.is_finite()),
            "speaker embedding contains non-finite values"
        );
        anyhow::ensure!(
            !self.model_id.is_empty(),
            "speaker embedding model is unnamed"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerObservation {
    pub segment_id: SegmentId,
    pub segment_sequence: u64,
    pub embedding: Option<SpeakerEmbedding>,
    /// Simultaneous embeddings, if an overlap-capable adapter supplies them.
    #[serde(default)]
    pub overlapping_embeddings: Vec<SpeakerEmbedding>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceFamiliarityEvidence {
    pub current_signature_id: VoiceSignatureId,
    pub prior_signature_id: VoiceSignatureId,
    pub similarity: f32,
    pub model_id: String,
    pub retention: VoiceRetentionScope,
}

impl VoiceFamiliarityEvidence {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.current_signature_id != self.prior_signature_id,
            "voice familiarity requires two distinct observations"
        );
        anyhow::ensure!(
            self.similarity.is_finite() && (-1.0..=1.0).contains(&self.similarity),
            "voice familiarity similarity is invalid"
        );
        anyhow::ensure!(
            !self.model_id.is_empty(),
            "voice familiarity model is unnamed"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnrolledSpeakerMapping {
    pub cluster_id: SpeakerClusterId,
    pub person_label: String,
    pub enrollment_id: String,
    pub consent_record: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum DiarizationEvent {
    SpeakerAssigned {
        segment_id: SegmentId,
        segment_sequence: u64,
        cluster_id: SpeakerClusterId,
        provisional: bool,
        similarity: Option<f32>,
        model_id: Option<String>,
    },
    SpeakerRevised {
        segment_id: SegmentId,
        segment_sequence: u64,
        from_cluster: SpeakerClusterId,
        to_cluster: SpeakerClusterId,
        reason: String,
    },
    Overlap {
        segment_id: SegmentId,
        segment_sequence: u64,
        clusters: Vec<SpeakerClusterId>,
    },
    UnknownSpeaker {
        segment_id: SegmentId,
        segment_sequence: u64,
        reason: String,
    },
    ClustersMerged {
        from_cluster: SpeakerClusterId,
        into_cluster: SpeakerClusterId,
    },
    VoiceFamiliarity {
        evidence: VoiceFamiliarityEvidence,
    },
    EnrolledPersonMapped {
        mapping: EnrolledSpeakerMapping,
    },
}

impl DiarizationEvent {
    /// Project without revising transcript text or sequence.
    pub fn stream_event(&self) -> StreamEvent {
        match self {
            Self::SpeakerAssigned {
                segment_id,
                cluster_id,
                similarity,
                ..
            } => StreamEvent::SpeakerAssigned {
                segment_id: segment_id.clone(),
                speaker_id: cluster_id.0.clone(),
                confidence: similarity.map(probability_confidence),
            },
            event => StreamEvent::DerivedArtifact {
                stage: "diarization".into(),
                artifact_id: event_artifact_id(event),
                value: serde_json::to_value(event).expect("diarization event is serializable"),
            },
        }
    }
}

pub trait SpeakerDiarizer {
    fn process(&mut self, observation: SpeakerObservation)
    -> anyhow::Result<Vec<DiarizationEvent>>;
    fn finish(&mut self) -> anyhow::Result<Vec<DiarizationEvent>>;
}

#[derive(Debug, Default)]
pub struct NoopSpeakerDiarizer;

impl SpeakerDiarizer for NoopSpeakerDiarizer {
    fn process(
        &mut self,
        _observation: SpeakerObservation,
    ) -> anyhow::Result<Vec<DiarizationEvent>> {
        Ok(Vec::new())
    }

    fn finish(&mut self) -> anyhow::Result<Vec<DiarizationEvent>> {
        Ok(Vec::new())
    }
}

pub fn diarize_offline(
    diarizer: &mut dyn SpeakerDiarizer,
    observations: impl IntoIterator<Item = SpeakerObservation>,
) -> anyhow::Result<Vec<DiarizationEvent>> {
    let mut events = Vec::new();
    for observation in observations {
        events.extend(diarizer.process(observation)?);
    }
    events.extend(diarizer.finish()?);
    Ok(events)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnonymousDiarizationConfig {
    pub match_threshold: f32,
    pub maximum_clusters: usize,
}

impl Default for AnonymousDiarizationConfig {
    fn default() -> Self {
        Self {
            match_threshold: 0.80,
            maximum_clusters: 16,
        }
    }
}

#[derive(Debug)]
struct AnonymousCluster {
    id: SpeakerClusterId,
    centroid: Vec<f32>,
    observations: u64,
    model_id: String,
}

pub struct AnonymousSpeakerClusterer {
    config: AnonymousDiarizationConfig,
    clusters: Vec<AnonymousCluster>,
    last_sequence: Option<u64>,
}

impl AnonymousSpeakerClusterer {
    pub fn new(config: AnonymousDiarizationConfig) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.match_threshold.is_finite() && (0.0..=1.0).contains(&config.match_threshold),
            "invalid speaker match threshold"
        );
        anyhow::ensure!(
            config.maximum_clusters > 0,
            "maximum_clusters must be positive"
        );
        Ok(Self {
            config,
            clusters: Vec::new(),
            last_sequence: None,
        })
    }

    pub fn merge(
        &mut self,
        from: &SpeakerClusterId,
        into: &SpeakerClusterId,
        affected: &[(SegmentId, u64)],
    ) -> anyhow::Result<Vec<DiarizationEvent>> {
        anyhow::ensure!(from != into, "cannot merge a speaker cluster into itself");
        let from_index = self
            .clusters
            .iter()
            .position(|cluster| &cluster.id == from)
            .ok_or_else(|| anyhow::anyhow!("source speaker cluster is unknown"))?;
        anyhow::ensure!(
            self.clusters.iter().any(|cluster| &cluster.id == into),
            "target speaker cluster is unknown"
        );
        self.clusters.remove(from_index);
        let mut events = vec![DiarizationEvent::ClustersMerged {
            from_cluster: from.clone(),
            into_cluster: into.clone(),
        }];
        events.extend(affected.iter().map(|(segment_id, sequence)| {
            DiarizationEvent::SpeakerRevised {
                segment_id: segment_id.clone(),
                segment_sequence: *sequence,
                from_cluster: from.clone(),
                to_cluster: into.clone(),
                reason: "clusters_merged".into(),
            }
        }));
        Ok(events)
    }

    fn assign(
        &mut self,
        embedding: SpeakerEmbedding,
    ) -> anyhow::Result<Option<(SpeakerClusterId, f32)>> {
        embedding.validate()?;
        if let Some(cluster) = self.clusters.first() {
            anyhow::ensure!(
                cluster.centroid.len() == embedding.values.len(),
                "speaker embedding dimensions changed within the stream"
            );
            anyhow::ensure!(
                cluster.model_id == embedding.model_id,
                "speaker embedding model changed within the stream"
            );
        }
        let best = self
            .clusters
            .iter()
            .enumerate()
            .map(|(index, cluster)| (index, cosine(&cluster.centroid, &embedding.values)))
            .max_by(|left, right| left.1.total_cmp(&right.1));
        if let Some((index, similarity)) = best
            && similarity >= self.config.match_threshold
        {
            update_centroid(&mut self.clusters[index], &embedding.values);
            return Ok(Some((self.clusters[index].id.clone(), similarity)));
        }
        if self.clusters.len() >= self.config.maximum_clusters {
            return Ok(None);
        }
        let id = SpeakerClusterId(format!("speaker:{}", self.clusters.len()));
        self.clusters.push(AnonymousCluster {
            id: id.clone(),
            centroid: normalized(embedding.values),
            observations: 1,
            model_id: embedding.model_id,
        });
        Ok(Some((id, 1.0)))
    }
}

impl SpeakerDiarizer for AnonymousSpeakerClusterer {
    fn process(
        &mut self,
        observation: SpeakerObservation,
    ) -> anyhow::Result<Vec<DiarizationEvent>> {
        if self
            .last_sequence
            .is_some_and(|previous| observation.segment_sequence <= previous)
        {
            anyhow::bail!("diarization received an out-of-order segment");
        }
        self.last_sequence = Some(observation.segment_sequence);
        let Some(embedding) = observation.embedding else {
            return Ok(vec![DiarizationEvent::UnknownSpeaker {
                segment_id: observation.segment_id,
                segment_sequence: observation.segment_sequence,
                reason: "no_embedding".into(),
            }]);
        };
        let model_id = embedding.model_id.clone();
        let Some((cluster_id, similarity)) = self.assign(embedding)? else {
            return Ok(vec![DiarizationEvent::UnknownSpeaker {
                segment_id: observation.segment_id,
                segment_sequence: observation.segment_sequence,
                reason: "cluster_capacity_exhausted".into(),
            }]);
        };
        let mut events = vec![DiarizationEvent::SpeakerAssigned {
            segment_id: observation.segment_id.clone(),
            segment_sequence: observation.segment_sequence,
            cluster_id: cluster_id.clone(),
            provisional: false,
            similarity: Some(similarity),
            model_id: Some(model_id),
        }];
        if !observation.overlapping_embeddings.is_empty() {
            let mut clusters = vec![cluster_id];
            for embedding in observation.overlapping_embeddings {
                if let Some((cluster_id, _)) = self.assign(embedding)? {
                    clusters.push(cluster_id);
                }
            }
            clusters.dedup();
            events.push(DiarizationEvent::Overlap {
                segment_id: observation.segment_id,
                segment_sequence: observation.segment_sequence,
                clusters,
            });
        }
        Ok(events)
    }

    fn finish(&mut self) -> anyhow::Result<Vec<DiarizationEvent>> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VoiceFamiliarityConfig {
    pub match_threshold: f32,
    pub retention: VoiceRetentionPolicy,
}

impl Default for VoiceFamiliarityConfig {
    fn default() -> Self {
        Self {
            match_threshold: 0.80,
            retention: VoiceRetentionPolicy::default(),
        }
    }
}

#[derive(Debug)]
struct FamiliarVoice {
    signature_id: VoiceSignatureId,
    embedding: Vec<f32>,
    model_id: String,
    retention: VoiceRetentionScope,
}

pub trait VoiceFamiliarityMatcher {
    fn observe(
        &mut self,
        signature_id: VoiceSignatureId,
        embedding: SpeakerEmbedding,
    ) -> anyhow::Result<Vec<VoiceFamiliarityEvidence>>;
    fn clear_session(&mut self);
}

#[derive(Debug)]
pub struct InMemoryVoiceFamiliarityMatcher {
    config: VoiceFamiliarityConfig,
    observations: Vec<FamiliarVoice>,
}

impl InMemoryVoiceFamiliarityMatcher {
    pub fn new(config: VoiceFamiliarityConfig) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.match_threshold.is_finite() && (-1.0..=1.0).contains(&config.match_threshold),
            "invalid voice familiarity threshold"
        );
        Ok(Self {
            config,
            observations: Vec::new(),
        })
    }
}

impl VoiceFamiliarityMatcher for InMemoryVoiceFamiliarityMatcher {
    fn observe(
        &mut self,
        signature_id: VoiceSignatureId,
        embedding: SpeakerEmbedding,
    ) -> anyhow::Result<Vec<VoiceFamiliarityEvidence>> {
        embedding.validate()?;
        anyhow::ensure!(
            self.config.retention.permits(embedding.retention),
            "voice embedding retention scope is not permitted"
        );

        let mut evidence = Vec::new();
        for prior in &self.observations {
            anyhow::ensure!(
                prior.embedding.len() == embedding.values.len(),
                "speaker embedding dimensions changed within familiarity scope"
            );
            if prior.model_id != embedding.model_id {
                continue;
            }
            let similarity = cosine(&prior.embedding, &embedding.values);
            if similarity >= self.config.match_threshold {
                let item = VoiceFamiliarityEvidence {
                    current_signature_id: signature_id.clone(),
                    prior_signature_id: prior.signature_id.clone(),
                    similarity,
                    model_id: embedding.model_id.clone(),
                    retention: prior.retention,
                };
                item.validate()?;
                evidence.push(item);
            }
        }

        if embedding.retention != VoiceRetentionScope::Segment {
            self.observations.push(FamiliarVoice {
                signature_id,
                embedding: normalized(embedding.values),
                model_id: embedding.model_id,
                retention: embedding.retention,
            });
        }
        Ok(evidence)
    }

    fn clear_session(&mut self) {
        self.observations
            .retain(|voice| voice.retention == VoiceRetentionScope::PersistentOptIn);
    }
}

#[derive(Debug, Default)]
pub struct DiarizationProjection {
    assignments: std::collections::HashMap<SegmentId, SpeakerClusterId>,
}

impl DiarizationProjection {
    pub fn observe(&mut self, event: &DiarizationEvent) {
        match event {
            DiarizationEvent::SpeakerAssigned {
                segment_id,
                cluster_id,
                ..
            } => {
                self.assignments
                    .insert(segment_id.clone(), cluster_id.clone());
            }
            DiarizationEvent::SpeakerRevised {
                segment_id,
                to_cluster,
                ..
            } => {
                self.assignments
                    .insert(segment_id.clone(), to_cluster.clone());
            }
            _ => {}
        }
    }

    pub fn project(&self, event: StreamEvent) -> StreamEvent {
        match event {
            StreamEvent::CommittedSegment {
                role,
                segment_id,
                text,
                words,
                language,
                speaker_id,
                confidence,
            } => StreamEvent::CommittedSegment {
                role,
                speaker_id: self
                    .assignments
                    .get(&segment_id)
                    .map(|speaker| speaker.0.clone())
                    .or(speaker_id),
                segment_id,
                text,
                words,
                language,
                confidence,
            },
            event => event,
        }
    }
}

fn update_centroid(cluster: &mut AnonymousCluster, observation: &[f32]) {
    let count = cluster.observations as f32;
    for (centroid, value) in cluster.centroid.iter_mut().zip(observation) {
        *centroid = (*centroid * count + *value) / (count + 1.0);
    }
    cluster.centroid = normalized(std::mem::take(&mut cluster.centroid));
    cluster.observations = cluster.observations.saturating_add(1);
}

fn normalized(mut values: Vec<f32>) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in &mut values {
            *value /= norm;
        }
    }
    values
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let left = normalized(left.to_vec());
    let right = normalized(right.to_vec());
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn probability_confidence(value: f32) -> Confidence {
    Confidence {
        value: f64::from(value.clamp(0.0, 1.0)),
        scale: ConfidenceScale::Probability,
        calibration: None,
    }
}

fn event_artifact_id(event: &DiarizationEvent) -> String {
    match event {
        DiarizationEvent::SpeakerRevised { segment_id, .. }
        | DiarizationEvent::Overlap { segment_id, .. }
        | DiarizationEvent::UnknownSpeaker { segment_id, .. } => segment_id.0.clone(),
        DiarizationEvent::ClustersMerged { from_cluster, .. } => from_cluster.0.clone(),
        DiarizationEvent::VoiceFamiliarity { evidence } => evidence.current_signature_id.0.clone(),
        DiarizationEvent::EnrolledPersonMapped { mapping } => mapping.enrollment_id.clone(),
        DiarizationEvent::SpeakerAssigned { segment_id, .. } => segment_id.0.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(values: &[f32]) -> SpeakerEmbedding {
        SpeakerEmbedding {
            values: values.to_vec(),
            model_id: "fixture-embedding".into(),
            retention: VoiceRetentionScope::Session,
        }
    }

    fn observation(sequence: u64, values: &[f32]) -> SpeakerObservation {
        SpeakerObservation {
            segment_id: SegmentId(format!("segment:{sequence}")),
            segment_sequence: sequence,
            embedding: Some(embedding(values)),
            overlapping_embeddings: Vec::new(),
        }
    }

    #[test]
    fn anonymous_clusters_need_no_enrollment_or_voiceprint_storage() {
        let mut diarizer =
            AnonymousSpeakerClusterer::new(AnonymousDiarizationConfig::default()).unwrap();
        let events = diarizer.process(observation(0, &[1.0, 0.0])).unwrap();
        assert!(matches!(
            &events[0],
            DiarizationEvent::SpeakerAssigned { cluster_id, .. } if cluster_id.0 == "speaker:0"
        ));
        assert!(
            !serde_json::to_string(&events)
                .unwrap()
                .contains("person_label")
        );
    }

    #[test]
    fn speaker_change_and_return_keep_stable_cluster_ids() {
        let mut diarizer =
            AnonymousSpeakerClusterer::new(AnonymousDiarizationConfig::default()).unwrap();
        let first = diarizer.process(observation(0, &[1.0, 0.0])).unwrap();
        let second = diarizer.process(observation(1, &[0.0, 1.0])).unwrap();
        let third = diarizer.process(observation(2, &[0.99, 0.01])).unwrap();
        assert!(matches!(
            &first[0],
            DiarizationEvent::SpeakerAssigned { cluster_id, .. } if cluster_id.0 == "speaker:0"
        ));
        assert!(matches!(
            &second[0],
            DiarizationEvent::SpeakerAssigned { cluster_id, .. } if cluster_id.0 == "speaker:1"
        ));
        assert!(matches!(
            &third[0],
            DiarizationEvent::SpeakerAssigned { cluster_id, .. } if cluster_id.0 == "speaker:0"
        ));
    }

    #[test]
    fn overlapping_speech_emits_multiple_anonymous_clusters() {
        let mut diarizer =
            AnonymousSpeakerClusterer::new(AnonymousDiarizationConfig::default()).unwrap();
        let mut overlap = observation(0, &[1.0, 0.0]);
        overlap.overlapping_embeddings = vec![embedding(&[0.0, 1.0])];
        let events = diarizer.process(overlap).unwrap();
        assert!(matches!(
            &events[1],
            DiarizationEvent::Overlap { clusters, .. } if clusters.len() == 2
        ));
    }

    #[test]
    fn revisions_preserve_original_segment_sequence() {
        let mut diarizer =
            AnonymousSpeakerClusterer::new(AnonymousDiarizationConfig::default()).unwrap();
        diarizer.process(observation(0, &[1.0, 0.0])).unwrap();
        diarizer.process(observation(1, &[0.0, 1.0])).unwrap();
        let revisions = diarizer
            .merge(
                &SpeakerClusterId("speaker:1".into()),
                &SpeakerClusterId("speaker:0".into()),
                &[(SegmentId("segment:1".into()), 1)],
            )
            .unwrap();
        assert!(matches!(
            &revisions[1],
            DiarizationEvent::SpeakerRevised {
                segment_sequence: 1,
                ..
            }
        ));
    }

    #[test]
    fn familiarity_evidence_cannot_be_serialized_as_person_identity() {
        let event = DiarizationEvent::VoiceFamiliarity {
            evidence: VoiceFamiliarityEvidence {
                current_signature_id: VoiceSignatureId("session:current".into()),
                prior_signature_id: VoiceSignatureId("session:prior".into()),
                similarity: 0.91,
                model_id: "fixture-embedding".into(),
                retention: VoiceRetentionScope::Session,
            },
        };
        let encoded = serde_json::to_string(&event).unwrap();
        assert!(encoded.contains("voice_familiarity"));
        assert!(!encoded.contains("person_label"));
    }

    #[test]
    fn noop_and_offline_paths_share_the_streaming_interface() {
        let observations = [observation(0, &[1.0, 0.0]), observation(1, &[0.0, 1.0])];
        assert!(
            diarize_offline(&mut NoopSpeakerDiarizer, observations.clone())
                .unwrap()
                .is_empty()
        );

        let mut diarizer =
            AnonymousSpeakerClusterer::new(AnonymousDiarizationConfig::default()).unwrap();
        let events = diarize_offline(&mut diarizer, observations).unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn cluster_capacity_exhaustion_is_unknown_not_an_identity() {
        let mut diarizer = AnonymousSpeakerClusterer::new(AnonymousDiarizationConfig {
            match_threshold: 0.8,
            maximum_clusters: 1,
        })
        .unwrap();
        diarizer.process(observation(0, &[1.0, 0.0])).unwrap();
        let events = diarizer.process(observation(1, &[0.0, 1.0])).unwrap();
        assert!(matches!(
            &events[0],
            DiarizationEvent::UnknownSpeaker { reason, .. }
                if reason == "cluster_capacity_exhausted"
        ));
    }

    #[test]
    fn embedding_models_cannot_change_silently_within_a_stream() {
        let mut diarizer =
            AnonymousSpeakerClusterer::new(AnonymousDiarizationConfig::default()).unwrap();
        diarizer.process(observation(0, &[1.0, 0.0])).unwrap();
        let mut changed = observation(1, &[1.0, 0.0]);
        changed.embedding.as_mut().unwrap().model_id = "other-model".into();
        assert!(
            diarizer
                .process(changed)
                .unwrap_err()
                .to_string()
                .contains("model changed")
        );
    }

    #[test]
    fn committed_transcript_projection_preserves_speaker_after_revision() {
        let segment_id = SegmentId("segment:7".into());
        let mut projection = DiarizationProjection::default();
        projection.observe(&DiarizationEvent::SpeakerAssigned {
            segment_id: segment_id.clone(),
            segment_sequence: 7,
            cluster_id: SpeakerClusterId("speaker:1".into()),
            provisional: true,
            similarity: Some(0.82),
            model_id: Some("fixture-embedding".into()),
        });
        projection.observe(&DiarizationEvent::SpeakerRevised {
            segment_id: segment_id.clone(),
            segment_sequence: 7,
            from_cluster: SpeakerClusterId("speaker:1".into()),
            to_cluster: SpeakerClusterId("speaker:0".into()),
            reason: "clusters_merged".into(),
        });

        let projected = projection.project(StreamEvent::CommittedSegment {
            role: crate::TextRole::Recognition,
            segment_id: segment_id.clone(),
            text: "hello".into(),
            words: Vec::new(),
            language: None,
            speaker_id: None,
            confidence: None,
        });
        assert!(matches!(
            projected,
            StreamEvent::CommittedSegment {
                segment_id: projected_id,
                speaker_id: Some(speaker),
                ..
            } if projected_id == segment_id && speaker == "speaker:0"
        ));
    }

    #[test]
    fn familiarity_operates_without_enrollment_or_names() {
        let mut matcher =
            InMemoryVoiceFamiliarityMatcher::new(VoiceFamiliarityConfig::default()).unwrap();
        assert!(
            matcher
                .observe(
                    VoiceSignatureId("voice:first".into()),
                    embedding(&[1.0, 0.0])
                )
                .unwrap()
                .is_empty()
        );
        let evidence = matcher
            .observe(
                VoiceSignatureId("voice:second".into()),
                embedding(&[0.99, 0.01]),
            )
            .unwrap();
        assert_eq!(evidence.len(), 1);
        let encoded = serde_json::to_string(&evidence).unwrap();
        assert!(encoded.contains("fixture-embedding"));
        assert!(!encoded.contains("person"));
        assert!(!encoded.contains("enrollment"));
    }

    #[test]
    fn persistent_voice_retention_requires_explicit_opt_in() {
        let mut matcher =
            InMemoryVoiceFamiliarityMatcher::new(VoiceFamiliarityConfig::default()).unwrap();
        let mut persistent = embedding(&[1.0, 0.0]);
        persistent.retention = VoiceRetentionScope::PersistentOptIn;
        assert!(
            matcher
                .observe(VoiceSignatureId("voice:persistent".into()), persistent)
                .unwrap_err()
                .to_string()
                .contains("not permitted")
        );

        let mut opted_in = InMemoryVoiceFamiliarityMatcher::new(VoiceFamiliarityConfig {
            match_threshold: 0.8,
            retention: VoiceRetentionPolicy {
                retain_for_session: true,
                retain_persistently: true,
            },
        })
        .unwrap();
        let mut persistent = embedding(&[1.0, 0.0]);
        persistent.retention = VoiceRetentionScope::PersistentOptIn;
        opted_in
            .observe(VoiceSignatureId("voice:persistent".into()), persistent)
            .unwrap();
        opted_in.clear_session();
        let mut next = embedding(&[1.0, 0.0]);
        next.retention = VoiceRetentionScope::PersistentOptIn;
        assert_eq!(
            opted_in
                .observe(VoiceSignatureId("voice:next".into()), next)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn clearing_a_session_forgets_session_scoped_familiarity() {
        let mut matcher =
            InMemoryVoiceFamiliarityMatcher::new(VoiceFamiliarityConfig::default()).unwrap();
        matcher
            .observe(
                VoiceSignatureId("voice:first".into()),
                embedding(&[1.0, 0.0]),
            )
            .unwrap();
        matcher.clear_session();
        assert!(
            matcher
                .observe(
                    VoiceSignatureId("voice:second".into()),
                    embedding(&[1.0, 0.0])
                )
                .unwrap()
                .is_empty()
        );
    }
}
