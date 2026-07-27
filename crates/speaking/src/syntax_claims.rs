use crate::event::TextRange;
use crate::evidence::{
    ClaimRationale, ClaimResolutionId, LinguisticClaim, LinguisticClaimError, LinguisticClaimId,
    LinguisticClaimKind, LinguisticClaimValue, LinguisticEvidenceArtifact, LinguisticTarget,
    LinguisticTargetScope,
};
use crate::ids::UtteranceId;
use crate::syntax::{
    GrammarAnalysis, GrammarAnalysisStatus, GrammarParseId, GrammarParseStatus,
    GrammarRankingPolicy, SyntacticLinkKind,
};

impl GrammarAnalysis {
    pub fn to_linguistic_evidence(
        &self,
        utterance_id: UtteranceId,
        word_ranges: Option<&[TextRange]>,
    ) -> Result<LinguisticEvidenceArtifact, LinguisticClaimError> {
        let mut artifact = LinguisticEvidenceArtifact::new(utterance_id.clone());
        let parse_claim_ids = self
            .ranked_parses
            .iter()
            .map(|parse| parse_claim_id(&parse.id))
            .collect::<Vec<_>>();

        for (parse_index, parse) in self.ranked_parses.iter().enumerate() {
            let parse_claim_id = parse_claim_ids[parse_index].clone();
            let token_facts = self
                .token_facts_for_parse(&parse.id)
                .unwrap_or_else(|| self.tokens.clone());
            let mut component_ids = Vec::new();

            for link in &parse.links {
                let claim_id = LinguisticClaimId(format!(
                    "{}:link:{}:{}:{}",
                    parse_claim_id.0,
                    link.left,
                    link.right,
                    link_kind_label(link.kind)
                ));
                let target_id = format!(
                    "link:{}:{}:{}",
                    link.left,
                    link.right,
                    link_kind_label(link.kind)
                );
                let target = LinguisticTarget::new(
                    utterance_id.clone(),
                    LinguisticTargetScope::SyntaxLink { id: target_id },
                    combined_range(word_ranges, link.left, link.right),
                );
                let claim = LinguisticClaim::grammar(
                    claim_id.clone(),
                    target,
                    LinguisticClaimValue::DependencyLink {
                        left: link.left,
                        right: link.right,
                        kind: link.kind,
                    },
                    claim_probability(link.confidence),
                    ClaimRationale::new(
                        "grammar.parse.link",
                        format!("link asserted by grammar parse {}", parse.id.0),
                    )
                    .with_attribute("parse_id", parse.id.0.clone())
                    .with_attribute("parse_rank", format!("{:.6}", parse.rank)),
                )?;
                artifact.insert_claim(claim)?;
                component_ids.push(claim_id);
            }

            for token in &token_facts {
                let pos_claim_id =
                    LinguisticClaimId(format!("{}:pos:{}", parse_claim_id.0, token.word_index));
                let target = LinguisticTarget::new(
                    utterance_id.clone(),
                    LinguisticTargetScope::Token {
                        id: format!("token:{}", token.word_index),
                    },
                    word_ranges.and_then(|ranges| ranges.get(token.word_index).cloned()),
                );
                let claim = LinguisticClaim::grammar(
                    pos_claim_id.clone(),
                    target.clone(),
                    LinguisticClaimValue::PartOfSpeech(token.pos),
                    claim_probability(parse.confidence),
                    ClaimRationale::new(
                        "grammar.parse.pos",
                        format!("part of speech asserted by grammar parse {}", parse.id.0),
                    )
                    .with_attribute("parse_id", parse.id.0.clone()),
                )?;
                artifact.insert_claim(claim)?;
                component_ids.push(pos_claim_id);

                let prosody_claim_id = LinguisticClaimId(format!(
                    "{}:prosodic-role:{}",
                    parse_claim_id.0, token.word_index
                ));
                let claim = LinguisticClaim::grammar(
                    prosody_claim_id.clone(),
                    target,
                    LinguisticClaimValue::ProsodicRole(token.prosodic_role),
                    claim_probability(parse.confidence),
                    ClaimRationale::new(
                        "grammar.parse.prosodic_role",
                        format!("prosodic role asserted by grammar parse {}", parse.id.0),
                    )
                    .with_attribute("parse_id", parse.id.0.clone()),
                )?;
                artifact.insert_claim(claim)?;
                component_ids.push(prosody_claim_id);
            }

            let mut parse_claim = LinguisticClaim::grammar(
                parse_claim_id.clone(),
                LinguisticTarget::parse(
                    utterance_id.clone(),
                    "grammar-selection",
                    full_range(word_ranges),
                ),
                LinguisticClaimValue::Parse {
                    parse_id: parse.id.0.clone(),
                },
                claim_probability(parse.rank),
                ClaimRationale::new(
                    "grammar.parse.candidate",
                    format!(
                        "normalized rank {:.6}, confidence {:.6}, variant {:?}",
                        parse.rank, parse.confidence, parse.provenance.variant
                    ),
                )
                .with_attribute("parse_id", parse.id.0.clone())
                .with_attribute(
                    "backend_parse_index",
                    parse.provenance.backend_parse_index.to_string(),
                ),
            )?;
            for support in component_ids {
                parse_claim = parse_claim.with_support(support);
            }
            for conflict in &parse_claim_ids {
                if conflict != &parse_claim_id {
                    parse_claim = parse_claim.with_conflict(conflict.clone());
                }
            }
            artifact.insert_claim(parse_claim)?;
        }

        if self.status == GrammarAnalysisStatus::Complete {
            let stable_ids = artifact
                .claims
                .iter()
                .filter(|claim| {
                    self.ranked_parses.iter().any(|parse| {
                        parse.status == GrammarParseStatus::Complete
                            && claim.id.0.starts_with(&parse_claim_id(&parse.id).0)
                    })
                })
                .map(|claim| claim.id.clone())
                .collect::<Vec<_>>();
            for id in stable_ids {
                artifact.stabilize_claim(&id, "complete grammar analysis")?;
            }
        }

        if !parse_claim_ids.is_empty() {
            artifact.resolve(
                ClaimResolutionId("resolution:grammar-selection".into()),
                &LinguisticTarget::parse(
                    utterance_id,
                    "grammar-selection",
                    full_range(word_ranges),
                ),
                LinguisticClaimKind::Parse,
            )?;
        }
        Ok(artifact)
    }

    pub fn conservative_linguistic_evidence(
        &self,
        utterance_id: UtteranceId,
        word_ranges: Option<&[TextRange]>,
        policy: GrammarRankingPolicy,
    ) -> Result<LinguisticEvidenceArtifact, LinguisticClaimError> {
        let mut artifact = self.to_linguistic_evidence(utterance_id, word_ranges)?;
        if !self.permits_irreversible_prosody(policy) {
            for resolution in &mut artifact.resolutions {
                for candidate in &mut resolution.candidates {
                    candidate.explanation.push_str(
                        "; interpretation remains conservative until rank/confidence margin clears",
                    );
                }
            }
        }
        Ok(artifact)
    }
}

fn parse_claim_id(parse_id: &GrammarParseId) -> LinguisticClaimId {
    LinguisticClaimId(format!("claim:{}", parse_id.0))
}

fn claim_probability(value: f32) -> f64 {
    f64::from((value.clamp(0.0, 1.0) * 1_000_000.0).round()) / 1_000_000.0
}

fn combined_range(
    word_ranges: Option<&[TextRange]>,
    left: usize,
    right: usize,
) -> Option<TextRange> {
    let ranges = word_ranges?;
    let left = ranges.get(left)?;
    let right = ranges.get(right)?;
    Some(TextRange {
        start: left.start.min(right.start),
        end: left.end.max(right.end),
    })
}

fn full_range(word_ranges: Option<&[TextRange]>) -> Option<TextRange> {
    let ranges = word_ranges?;
    Some(TextRange {
        start: ranges.first()?.start,
        end: ranges.last()?.end,
    })
}

fn link_kind_label(kind: SyntacticLinkKind) -> &'static str {
    match kind {
        SyntacticLinkKind::Subject => "subject",
        SyntacticLinkKind::Object => "object",
        SyntacticLinkKind::Complement => "complement",
        SyntacticLinkKind::InfinitivalMarker => "infinitival-marker",
        SyntacticLinkKind::Modifier => "modifier",
        SyntacticLinkKind::Determiner => "determiner",
        SyntacticLinkKind::Auxiliary => "auxiliary",
        SyntacticLinkKind::Preposition => "preposition",
        SyntacticLinkKind::Coordination => "coordination",
        SyntacticLinkKind::ContrastPair => "contrast-pair",
        SyntacticLinkKind::NounCompound => "noun-compound",
        SyntacticLinkKind::Vocative => "vocative",
        SyntacticLinkKind::Apposition => "apposition",
        SyntacticLinkKind::Parenthetical => "parenthetical",
    }
}
