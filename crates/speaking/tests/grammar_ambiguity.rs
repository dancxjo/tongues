use speaking::{
    GrammarAnalysisStatus, GrammarBackend, GrammarInterpretationMode, GrammarParseStatus,
    GrammarParseVariant, GrammarParser, GrammarRankingPolicy, LinguisticClaimKind,
    LinguisticEvidenceArtifact, PartOfSpeech, SyntacticLinkKind, TextRange, UtteranceId,
    VarietyGrammarParser, VarietyId,
};

type ParseVariantMatcher = fn(&GrammarParseVariant) -> bool;

fn words(sentence: &str) -> Vec<String> {
    sentence
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|character: char| !character.is_alphabetic() && character != '\'')
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn parse(sentence: &str) -> speaking::GrammarAnalysis {
    VarietyGrammarParser::new(VarietyId("en-US-GA".into())).parse(&words(sentence), None)
}

#[test]
fn pp_attachment_is_retained_ranked_and_conservative() {
    let analysis = parse("I saw the man with the telescope.");
    assert_eq!(analysis.status, GrammarAnalysisStatus::Complete);
    assert_eq!(analysis.ranked_parses.len(), 3);
    assert!(analysis.ranked_parses.windows(2).all(|pair| {
        pair[0].rank > pair[1].rank || (pair[0].rank == pair[1].rank && pair[0].id < pair[1].id)
    }));
    assert_eq!(
        analysis.interpretation_mode(GrammarRankingPolicy::default()),
        GrammarInterpretationMode::Conservative
    );
    assert!(!analysis.permits_irreversible_prosody(GrammarRankingPolicy::default()));

    let pp_variants = analysis
        .ranked_parses
        .iter()
        .filter(|parse| {
            matches!(
                parse.provenance.variant,
                GrammarParseVariant::PrepositionalAttachment { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(pp_variants.len(), 2);
    assert!(
        pp_variants
            .iter()
            .any(|parse| parse.links.iter().any(|link| {
                link.left == 3 && link.right == 6 && link.kind == SyntacticLinkKind::Modifier
            }))
    );
    assert!(
        pp_variants
            .iter()
            .any(|parse| parse.links.iter().any(|link| {
                link.left == 1 && link.right == 6 && link.kind == SyntacticLinkKind::Complement
            }))
    );

    let conservative = analysis.conservative_facts(GrammarRankingPolicy::default());
    assert!(conservative.suppress_irreversible_prosody);
    assert!(
        !conservative.tokens[6]
            .syntactic_links
            .contains(&SyntacticLinkKind::Modifier)
    );
    assert!(
        !conservative.tokens[6]
            .syntactic_links
            .contains(&SyntacticLinkKind::Complement)
    );
}

#[test]
fn unambiguous_fixture_does_not_grow_alternatives() {
    let analysis = parse("I saw the man.");
    assert_eq!(analysis.ranked_parses.len(), 1);
    assert!(analysis.alternatives().is_empty());
    assert_eq!(
        analysis.interpretation_mode(GrammarRankingPolicy::default()),
        GrammarInterpretationMode::Decisive
    );
}

#[test]
fn native_rules_cover_bounded_ambiguity_families() {
    let fixtures: [(&str, ParseVariantMatcher); 6] = [
        ("old men and women", |variant: &GrammarParseVariant| {
            matches!(variant, GrammarParseVariant::CoordinationScope { .. })
        }),
        ("I saw her record", |variant: &GrammarParseVariant| {
            matches!(
                variant,
                GrammarParseVariant::PartOfSpeech {
                    pos: PartOfSpeech::Verb,
                    ..
                }
            )
        }),
        (
            "I said I know that she left",
            |variant: &GrammarParseVariant| {
                matches!(variant, GrammarParseVariant::ComplementAttachment { .. })
            },
        ),
        (
            "they turn up the volume",
            |variant: &GrammarParseVariant| {
                matches!(variant, GrammarParseVariant::PhrasalParticle { .. })
            },
        ),
        (
            "I saw the picture of the man who smiled",
            |variant: &GrammarParseVariant| {
                matches!(variant, GrammarParseVariant::RelativeClause { .. })
            },
        ),
        ("they apparently work", |variant: &GrammarParseVariant| {
            matches!(variant, GrammarParseVariant::PunctuationIsland { .. })
        }),
    ];

    for (sentence, expected) in fixtures {
        let analysis = parse(sentence);
        assert!(
            analysis
                .alternatives()
                .iter()
                .any(|parse| expected(&parse.provenance.variant)),
            "missing expected ambiguity family for {sentence:?}: {analysis:#?}"
        );
        assert!(
            analysis.ranked_parses.len() <= speaking::DEFAULT_MAX_GRAMMAR_ALTERNATIVES,
            "alternative cap exceeded for {sentence:?}"
        );
    }
}

#[test]
fn parse_ids_survive_extension_and_revision_delta_is_explicit() {
    let prefix = parse("I saw the man with the telescope");
    let extended = parse("I saw the man with the telescope yesterday");
    let delta = extended.identity_delta_from(&prefix);
    assert!(delta.retained.contains(&prefix.best_parse().unwrap().id));
    assert!(delta.retained.iter().any(|id| id.0.contains(":pp-")));

    let repaired = parse("I saw the woman");
    let repair_delta = repaired.identity_delta_from(&extended);
    assert!(
        repair_delta
            .invalidated
            .iter()
            .any(|id| id.0.contains(":pp-"))
    );
}

#[test]
fn alternative_generation_is_bounded_for_large_inputs() {
    let sentence = std::iter::repeat_n("word", 200)
        .collect::<Vec<_>>()
        .join(" ");
    let analysis = parse(&sentence);
    assert_eq!(analysis.ranked_parses.len(), 1);
}

#[test]
fn failed_and_partial_analysis_are_not_empty_accepted_successes() {
    let failed =
        VarietyGrammarParser::new(VarietyId("not-a-variety".into())).parse(&words("hello"), None);
    assert_eq!(failed.status, GrammarAnalysisStatus::Failed);
    assert!(failed.ranked_parses.is_empty());
    assert!(failed.diagnostic.is_some());

    let partial = parse("xyzzy");
    assert_eq!(partial.status, GrammarAnalysisStatus::Partial);
    assert_eq!(partial.ranked_parses.len(), 1);
    assert_eq!(partial.ranked_parses[0].status, GrammarParseStatus::Partial);
    assert!(!partial.backend_parses[0].accepted);
}

#[test]
fn every_parse_emits_supported_claims_and_round_trips() {
    let analysis = parse("I saw the man with the telescope.");
    let ranges = words("I saw the man with the telescope.")
        .iter()
        .scan(0_u32, |start, word| {
            let range = TextRange {
                start: *start,
                end: *start + word.chars().count() as u32,
            };
            *start = range.end + 1;
            Some(range)
        })
        .collect::<Vec<_>>();
    let artifact = analysis
        .to_linguistic_evidence(UtteranceId("utt-grammar-ambiguity".into()), Some(&ranges))
        .unwrap();
    let parse_claims = artifact
        .claims
        .iter()
        .filter(|claim| claim.kind == LinguisticClaimKind::Parse)
        .collect::<Vec<_>>();
    assert_eq!(parse_claims.len(), analysis.ranked_parses.len());
    assert!(parse_claims.iter().all(|claim| !claim.supports.is_empty()));
    assert!(
        parse_claims
            .iter()
            .all(|claim| { claim.conflicts_with.len() + 1 == analysis.ranked_parses.len() })
    );
    assert_eq!(
        artifact.resolutions[0].winner.as_ref().unwrap().0,
        format!("claim:{}", analysis.best_parse().unwrap().id.0)
    );

    let json = artifact.to_json_pretty().unwrap();
    let decoded = LinguisticEvidenceArtifact::from_json_str(&json).unwrap();
    assert_eq!(decoded.schema_version, artifact.schema_version);
    assert_eq!(decoded.utterance_id, artifact.utterance_id);
    assert_eq!(decoded.claims.len(), artifact.claims.len());
    for (index, (decoded, original)) in decoded.claims.iter().zip(&artifact.claims).enumerate() {
        assert_eq!(decoded, original, "claim {index}");
    }
    assert_eq!(decoded.lifecycle, artifact.lifecycle);
    assert_eq!(decoded.resolutions, artifact.resolutions);
    assert_eq!(
        analysis.best_parse().unwrap().provenance.backend,
        GrammarBackend::TonguesRules
    );
    assert!(analysis.backend_parses[0].cost.is_some());
}
