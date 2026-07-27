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

    fn assign(&mut self, embedding: SpeakerEmbedding) -> anyhow::Result<(SpeakerClusterId, f32)> {
        embedding.validate()?;
        if self
            .clusters
            .first()
            .is_some_and(|cluster| cluster.centroid.len() != embedding.values.len())
        {
            anyhow::bail!("speaker embedding dimensions changed within the stream");
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
            return Ok((self.clusters[index].id.clone(), similarity));
        }
        if self.clusters.len() >= self.config.maximum_clusters {
            return Ok((SpeakerClusterId("speaker:unknown".into()), 0.0));
        }
        let id = SpeakerClusterId(format!("speaker:{}", self.clusters.len()));
        self.clusters.push(AnonymousCluster {
            id: id.clone(),
            centroid: normalized(embedding.values),
            observations: 1,
        });
        Ok((id, 1.0))
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
        let (cluster_id, similarity) = self.assign(embedding)?;
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
                clusters.push(self.assign(embedding)?.0);
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
}
