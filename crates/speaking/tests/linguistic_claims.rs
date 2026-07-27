use speaking::{
    ClaimConfidence, ClaimLifecycle, ClaimRationale, ClaimResolutionId, ClaimResolutionReason,
    EvidenceSource, LinguisticClaim, LinguisticClaimError, LinguisticClaimId, LinguisticClaimKind,
    LinguisticClaimValue, LinguisticEvidenceArtifact, LinguisticTarget, PartOfSpeech, StreamEvent,
    TextRange, UtteranceId, source_default_priority,
};

fn utterance() -> UtteranceId {
    UtteranceId("utt-claim-test".into())
}

fn target() -> LinguisticTarget {
    LinguisticTarget::word(utterance(), "word-1", TextRange { start: 4, end: 8 })
}

fn rationale(code: &str) -> ClaimRationale {
    ClaimRationale::new(code, format!("reason for {code}"))
}

fn grammar_claim(id: &str, pos: PartOfSpeech, probability: f64) -> LinguisticClaim {
    LinguisticClaim::grammar(
        LinguisticClaimId(id.into()),
        target(),
        LinguisticClaimValue::PartOfSpeech(pos),
        probability,
        rationale("grammar.pos"),
    )
    .unwrap()
}

#[test]
fn source_priority_orders_user_and_automatic_evidence() {
    assert!(
        source_default_priority(&EvidenceSource::ManualOverride)
            > source_default_priority(&EvidenceSource::UserMarkup)
    );
    assert!(
        source_default_priority(&EvidenceSource::UserMarkup)
            > source_default_priority(&EvidenceSource::CommittedAcoustics)
    );
    assert!(
        source_default_priority(&EvidenceSource::CommittedAcoustics)
            > source_default_priority(&EvidenceSource::Lexicon)
    );
    assert!(
        source_default_priority(&EvidenceSource::Lexicon)
            > source_default_priority(&EvidenceSource::Grammar)
    );
    assert!(
        source_default_priority(&EvidenceSource::Grammar)
            > source_default_priority(&EvidenceSource::Morphology)
    );
    assert!(
        source_default_priority(&EvidenceSource::Morphology)
            > source_default_priority(&EvidenceSource::Prosody)
    );
    assert!(
        source_default_priority(&EvidenceSource::Prosody)
            > source_default_priority(&EvidenceSource::Punctuation)
    );
    assert!(
        source_default_priority(&EvidenceSource::Punctuation)
            > source_default_priority(&EvidenceSource::ImportedData)
    );
    assert!(
        source_default_priority(&EvidenceSource::ImportedData)
            > source_default_priority(&EvidenceSource::LearnedPrediction)
    );
}

#[test]
fn manual_override_wins_without_erasing_automatic_claim() {
    let automatic = grammar_claim("automatic", PartOfSpeech::Noun, 0.99);
    let manual = LinguisticClaim::manual_override(
        LinguisticClaimId("manual".into()),
        target(),
        LinguisticClaimValue::PartOfSpeech(PartOfSpeech::Verb),
        rationale("user.override"),
    )
    .unwrap()
    .with_conflict(automatic.id.clone());

    let mut artifact = LinguisticEvidenceArtifact::new(utterance());
    artifact.insert_claim(automatic.clone()).unwrap();
    artifact.insert_claim(manual.clone()).unwrap();
    let resolution = artifact
        .resolve(
            ClaimResolutionId("resolution-pos".into()),
            &target(),
            LinguisticClaimKind::PartOfSpeech,
        )
        .unwrap();

    assert_eq!(resolution.winner, Some(manual.id));
    assert_eq!(artifact.claims.len(), 2);
    assert!(resolution.candidates.iter().any(|candidate| {
        candidate.claim_id == automatic.id
            && candidate.conflicts_with_winner
            && matches!(
                candidate.reason,
                ClaimResolutionReason::LowerPriority { .. }
            )
    }));
}

#[test]
fn equal_priority_uses_confidence_support_lifecycle_then_stable_id() {
    let support = grammar_claim("support", PartOfSpeech::Verb, 0.4);
    let first = grammar_claim("a", PartOfSpeech::Noun, 0.8);
    let second = grammar_claim("b", PartOfSpeech::Verb, 0.8).with_support(support.id.clone());
    let mut artifact = LinguisticEvidenceArtifact::new(utterance());
    artifact.insert_claim(first).unwrap();
    artifact.insert_claim(second.clone()).unwrap();
    artifact.insert_claim(support).unwrap();

    let resolution = artifact
        .resolve(
            ClaimResolutionId("support-tie".into()),
            &target(),
            LinguisticClaimKind::PartOfSpeech,
        )
        .unwrap();
    assert_eq!(resolution.winner, Some(second.id));
}

#[test]
fn resolution_is_deterministic_across_insertion_order() {
    let claims = [
        grammar_claim("c", PartOfSpeech::Noun, 0.7),
        grammar_claim("a", PartOfSpeech::Verb, 0.7),
        grammar_claim("b", PartOfSpeech::Adjective, 0.7),
    ];
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for (index, order) in orders.into_iter().enumerate() {
        let mut artifact = LinguisticEvidenceArtifact::new(utterance());
        for claim_index in order {
            artifact.insert_claim(claims[claim_index].clone()).unwrap();
        }
        let resolution = artifact
            .resolve(
                ClaimResolutionId(format!("permutation-{index}")),
                &target(),
                LinguisticClaimKind::PartOfSpeech,
            )
            .unwrap();
        assert_eq!(
            resolution.winner,
            Some(LinguisticClaimId("a".into())),
            "insertion order {order:?}"
        );
    }
}

#[test]
fn invalidated_and_revised_claims_never_win_but_remain_visible() {
    let stale = grammar_claim("stale", PartOfSpeech::Noun, 1.0);
    let replacement = grammar_claim("replacement", PartOfSpeech::Verb, 0.5);
    let invalid = grammar_claim("invalid", PartOfSpeech::Adjective, 1.0);
    let mut artifact = LinguisticEvidenceArtifact::new(utterance());
    artifact.insert_claim(stale.clone()).unwrap();
    artifact
        .revise_claim(&stale.id, replacement.clone(), "more context")
        .unwrap();
    artifact.insert_claim(invalid.clone()).unwrap();
    artifact
        .transition_claim(
            &invalid.id,
            ClaimLifecycle::Invalidated,
            "span replaced",
            None,
        )
        .unwrap();

    let resolution = artifact
        .resolve(
            ClaimResolutionId("revision".into()),
            &target(),
            LinguisticClaimKind::PartOfSpeech,
        )
        .unwrap();
    assert_eq!(resolution.winner, Some(replacement.id));
    assert_eq!(resolution.candidates.len(), 3);
    assert!(artifact.claim(&stale.id).is_some());
    assert!(artifact.claim(&invalid.id).is_some());
}

#[test]
fn failed_revision_is_transactional() {
    let stale = grammar_claim("stale", PartOfSpeech::Noun, 1.0);
    let replacement = grammar_claim("replacement", PartOfSpeech::Verb, 0.5);
    let mut artifact = LinguisticEvidenceArtifact::new(utterance());
    artifact.insert_claim(stale.clone()).unwrap();
    artifact
        .transition_claim(
            &stale.id,
            ClaimLifecycle::Invalidated,
            "evidence withdrawn",
            None,
        )
        .unwrap();

    assert!(matches!(
        artifact.revise_claim(&stale.id, replacement.clone(), "too late"),
        Err(LinguisticClaimError::InvalidLifecycleTransition { .. })
    ));
    assert!(artifact.claim(&replacement.id).is_none());
}

#[test]
fn text_revision_keeps_stable_prefix_identity_and_invalidates_tail_only() {
    let prefix_target =
        LinguisticTarget::word(utterance(), "prefix", TextRange { start: 0, end: 4 });
    let tail_target = LinguisticTarget::word(utterance(), "tail", TextRange { start: 4, end: 8 });
    let prefix = LinguisticClaim::grammar(
        LinguisticClaimId("prefix-claim".into()),
        prefix_target,
        LinguisticClaimValue::PartOfSpeech(PartOfSpeech::Pronoun),
        0.9,
        rationale("prefix"),
    )
    .unwrap();
    let tail = LinguisticClaim::grammar(
        LinguisticClaimId("tail-claim".into()),
        tail_target,
        LinguisticClaimValue::PartOfSpeech(PartOfSpeech::Verb),
        0.9,
        rationale("tail"),
    )
    .unwrap();
    let mut artifact = LinguisticEvidenceArtifact::new(utterance());
    artifact.insert_claim(prefix.clone()).unwrap();
    artifact.insert_claim(tail.clone()).unwrap();
    artifact
        .stabilize_claim(&prefix.id, "prefix unchanged")
        .unwrap();
    artifact
        .stabilize_claim(&tail.id, "tail initially stable")
        .unwrap();

    let before_failed_revision = artifact.clone();
    assert!(matches!(
        artifact.invalidate_text_revision(TextRange { start: 4, end: 8 }, " "),
        Err(LinguisticClaimError::MissingTransitionReason(_))
    ));
    assert_eq!(artifact, before_failed_revision);

    let invalidated = artifact
        .invalidate_text_revision(TextRange { start: 4, end: 8 }, "tail repaired")
        .unwrap();

    assert_eq!(invalidated, vec![tail.id.clone()]);
    assert_eq!(
        artifact.claim(&prefix.id).unwrap().lifecycle,
        ClaimLifecycle::Stable
    );
    assert_eq!(
        artifact.claim(&tail.id).unwrap().lifecycle,
        ClaimLifecycle::Invalidated
    );
}

#[test]
fn committed_claims_are_locked_and_stay_selected() {
    let committed = grammar_claim("committed", PartOfSpeech::Noun, 0.6);
    let manual = LinguisticClaim::manual_override(
        LinguisticClaimId("late-manual".into()),
        target(),
        LinguisticClaimValue::PartOfSpeech(PartOfSpeech::Verb),
        rationale("late.manual"),
    )
    .unwrap();
    let mut artifact = LinguisticEvidenceArtifact::new(utterance());
    artifact.insert_claim(committed.clone()).unwrap();
    artifact
        .commit_claim(&committed.id, "crossed commit frontier")
        .unwrap();
    artifact.insert_claim(manual).unwrap();

    let resolution = artifact
        .resolve(
            ClaimResolutionId("committed-resolution".into()),
            &target(),
            LinguisticClaimKind::PartOfSpeech,
        )
        .unwrap();
    assert_eq!(resolution.winner, Some(committed.id.clone()));
    assert!(matches!(
        artifact.invalidate_text_revision(TextRange { start: 4, end: 8 }, "illegal late repair"),
        Err(LinguisticClaimError::CommittedClaimCannotChange(id)) if id == committed.id
    ));
}

#[test]
fn claims_resolutions_and_edges_round_trip_through_event_artifact() {
    let first = grammar_claim("first", PartOfSpeech::Noun, 0.7);
    let second = grammar_claim("second", PartOfSpeech::Verb, 0.8).with_conflict(first.id.clone());
    let second_id = second.id.clone();
    let mut artifact = LinguisticEvidenceArtifact::new(utterance());
    artifact.insert_claim(first).unwrap();
    artifact.insert_claim(second).unwrap();
    artifact
        .stabilize_claim(&second_id, "winner survived the stable prefix")
        .unwrap();
    artifact
        .resolve(
            ClaimResolutionId("round-trip".into()),
            &target(),
            LinguisticClaimKind::PartOfSpeech,
        )
        .unwrap();

    let json = artifact.to_json_pretty().unwrap();
    let decoded = LinguisticEvidenceArtifact::from_json_str(&json).unwrap();
    assert_eq!(decoded, artifact);

    let event = artifact
        .as_derived_artifact("claims:utt-claim-test")
        .unwrap();
    let event_json = serde_json::to_string(&event).unwrap();
    let decoded_event: StreamEvent = serde_json::from_str(&event_json).unwrap();
    assert_eq!(decoded_event, event);
    assert_eq!(
        LinguisticEvidenceArtifact::from_derived_artifact(&decoded_event).unwrap(),
        artifact
    );
}

#[test]
fn unsupported_schema_and_invalid_confidence_fail_precisely() {
    assert!(matches!(
        ClaimConfidence::new(f64::NAN, None),
        Err(LinguisticClaimError::InvalidConfidence(value)) if value.is_nan()
    ));
    assert!(matches!(
        ClaimConfidence::new(-0.01, None),
        Err(LinguisticClaimError::InvalidConfidence(value)) if value == -0.01
    ));
    assert!(matches!(
        ClaimConfidence::new(1.01, None),
        Err(LinguisticClaimError::InvalidConfidence(value)) if value == 1.01
    ));

    let json =
        r#"{"schema_version":2,"utterance_id":"utt","claims":[],"lifecycle":[],"resolutions":[]}"#;
    assert_eq!(
        LinguisticEvidenceArtifact::from_json_str(json).unwrap_err(),
        LinguisticClaimError::UnsupportedSchema {
            found: 2,
            expected: 1
        }
    );
}
