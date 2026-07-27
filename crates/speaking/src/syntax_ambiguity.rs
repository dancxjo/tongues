use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::syntax::{
    GrammarBackend, GrammarParseId, GrammarParseProvenance, GrammarParseStatus,
    GrammarParseVariant, GrammarRankingPolicy, GrammarRuleSet, RankedGrammarParse, SyntacticLink,
    SyntacticLinkKind, SyntacticLinkSource, WordIndex,
};

pub(crate) fn rank_and_expand_parses(
    words: &[String],
    links: Vec<SyntacticLink>,
    backend: GrammarBackend,
    backend_parse_index: usize,
    profile: Option<GrammarRuleSet>,
    status: GrammarParseStatus,
    policy: GrammarRankingPolicy,
) -> Vec<RankedGrammarParse> {
    let mut base_links = links;
    canonicalize_links(&mut base_links);
    let confidence = normalized_confidence(&base_links);
    let rank = normalized_rank(&base_links, words.len(), status);
    let primary = RankedGrammarParse {
        id: parse_id(
            backend,
            backend_parse_index,
            &GrammarParseVariant::BackendPrimary,
        ),
        links: base_links,
        rank,
        confidence,
        status,
        provenance: GrammarParseProvenance {
            backend,
            backend_parse_index,
            variant: GrammarParseVariant::BackendPrimary,
        },
    };
    if words.len() > policy.max_tokens_for_alternatives || policy.max_alternatives <= 1 {
        return vec![primary];
    }

    let mut candidates = vec![primary.clone()];
    add_prepositional_attachment_variants(
        &mut candidates,
        &primary,
        words,
        backend,
        backend_parse_index,
        policy.max_alternatives,
    );
    add_coordination_scope_variants(
        &mut candidates,
        &primary,
        words,
        profile,
        backend,
        backend_parse_index,
        policy.max_alternatives,
    );
    if let Some(profile) = profile {
        add_pos_variants(
            &mut candidates,
            &primary,
            words,
            profile,
            backend,
            backend_parse_index,
            policy.max_alternatives,
        );
        add_phrasal_particle_variants(
            &mut candidates,
            &primary,
            words,
            profile,
            backend,
            backend_parse_index,
            policy.max_alternatives,
        );
        add_punctuation_island_variants(
            &mut candidates,
            &primary,
            words,
            profile,
            backend,
            backend_parse_index,
            policy.max_alternatives,
        );
    }
    add_complement_attachment_variants(
        &mut candidates,
        &primary,
        words,
        backend,
        backend_parse_index,
        policy.max_alternatives,
    );
    add_relative_clause_variants(
        &mut candidates,
        &primary,
        backend,
        backend_parse_index,
        policy.max_alternatives,
    );

    candidates.sort_by(compare_ranked_parses);
    candidates.truncate(policy.max_alternatives);
    candidates
}

fn add_prepositional_attachment_variants(
    candidates: &mut Vec<RankedGrammarParse>,
    primary: &RankedGrammarParse,
    words: &[String],
    backend: GrammarBackend,
    backend_parse_index: usize,
    limit: usize,
) {
    for preposition_link in primary
        .links
        .iter()
        .filter(|link| link.kind == SyntacticLinkKind::Preposition)
    {
        let preposition = preposition_link.left;
        let object = preposition_link.right;
        if preposition >= object || object >= words.len() {
            continue;
        }
        let verb = (0..preposition)
            .rev()
            .find(|index| is_predicate(*index, &primary.links));
        let noun = (0..preposition)
            .rev()
            .find(|index| Some(*index) != verb && is_nominal(*index, &primary.links));
        let (Some(verb), Some(noun)) = (verb, noun) else {
            continue;
        };

        let noun_variant = GrammarParseVariant::PrepositionalAttachment {
            preposition,
            object,
            head: noun,
        };
        let mut noun_links = primary.links.clone();
        push_variant_link(
            &mut noun_links,
            noun,
            object,
            SyntacticLinkKind::Modifier,
            0.46,
        );
        add_candidate(
            candidates,
            primary,
            noun_links,
            noun_variant,
            0.02,
            backend,
            backend_parse_index,
            limit,
        );

        let verb_variant = GrammarParseVariant::PrepositionalAttachment {
            preposition,
            object,
            head: verb,
        };
        let mut verb_links = primary.links.clone();
        push_variant_link(
            &mut verb_links,
            verb,
            object,
            SyntacticLinkKind::Complement,
            0.44,
        );
        add_candidate(
            candidates,
            primary,
            verb_links,
            verb_variant,
            0.04,
            backend,
            backend_parse_index,
            limit,
        );
        if candidates.len() >= limit {
            return;
        }
    }
}

fn add_coordination_scope_variants(
    candidates: &mut Vec<RankedGrammarParse>,
    primary: &RankedGrammarParse,
    words: &[String],
    profile: Option<GrammarRuleSet>,
    backend: GrammarBackend,
    backend_parse_index: usize,
    limit: usize,
) {
    let conjunctions = coordination_indices(words, &primary.links, profile);
    for conjunction in conjunctions {
        if conjunction == 0 || conjunction + 1 >= words.len() {
            continue;
        }
        let left = conjunction - 1;
        let right = conjunction + 1;
        let Some(scope_link) = primary
            .links
            .iter()
            .find(|link| link.right == left && link.kind == SyntacticLinkKind::Modifier)
        else {
            continue;
        };
        let variant = GrammarParseVariant::CoordinationScope {
            conjunction,
            left: scope_link.left,
            right,
        };
        let mut links = primary.links.clone();
        push_variant_link(&mut links, scope_link.left, right, scope_link.kind, 0.58);
        add_candidate(
            candidates,
            primary,
            links,
            variant,
            0.05,
            backend,
            backend_parse_index,
            limit,
        );
        if candidates.len() >= limit {
            return;
        }
    }
}

fn add_pos_variants(
    candidates: &mut Vec<RankedGrammarParse>,
    primary: &RankedGrammarParse,
    words: &[String],
    profile: GrammarRuleSet,
    backend: GrammarBackend,
    backend_parse_index: usize,
    limit: usize,
) {
    for word in 0..words.len() {
        let recognized_verb = profile.verbs.contains(&words[word].as_str())
            || profile
                .verb_suffixes
                .iter()
                .any(|suffix| words[word].ends_with(suffix));
        let determiner = primary
            .links
            .iter()
            .find(|link| link.right == word && link.kind == SyntacticLinkKind::Determiner);
        let ambiguous_possessive = determiner
            .is_some_and(|link| profile.object_pronouns.contains(&words[link.left].as_str()));
        if !recognized_verb || !ambiguous_possessive {
            continue;
        }
        let Some(predicate) = (0..word)
            .rev()
            .find(|index| is_predicate(*index, &primary.links))
        else {
            continue;
        };
        let mut links = primary.links.clone();
        links.retain(|link| !(link.right == word && link.kind == SyntacticLinkKind::Determiner));
        push_variant_link(
            &mut links,
            predicate,
            word,
            SyntacticLinkKind::Complement,
            0.51,
        );
        add_candidate(
            candidates,
            primary,
            links,
            GrammarParseVariant::PartOfSpeech {
                word,
                pos: crate::syntax::PartOfSpeech::Verb,
            },
            0.07,
            backend,
            backend_parse_index,
            limit,
        );
        if candidates.len() >= limit {
            return;
        }
    }
}

fn add_complement_attachment_variants(
    candidates: &mut Vec<RankedGrammarParse>,
    primary: &RankedGrammarParse,
    words: &[String],
    backend: GrammarBackend,
    backend_parse_index: usize,
    limit: usize,
) {
    for marker in 1..words.len() {
        let incoming = primary
            .links
            .iter()
            .filter(|link| link.right == marker && link.kind == SyntacticLinkKind::Complement)
            .map(|link| link.left)
            .collect::<BTreeSet<_>>();
        if incoming.is_empty()
            || !primary.links.iter().any(|link| {
                link.left == marker
                    && matches!(
                        link.kind,
                        SyntacticLinkKind::Complement | SyntacticLinkKind::Subject
                    )
            })
        {
            continue;
        }
        if incoming.len() > 1 {
            for head in incoming.iter().copied() {
                let mut links = primary.links.clone();
                links.retain(|link| {
                    !(link.right == marker && link.kind == SyntacticLinkKind::Complement)
                });
                push_variant_link(
                    &mut links,
                    head,
                    marker,
                    SyntacticLinkKind::Complement,
                    0.52,
                );
                add_candidate(
                    candidates,
                    primary,
                    links,
                    GrammarParseVariant::ComplementAttachment { marker, head },
                    0.05,
                    backend,
                    backend_parse_index,
                    limit,
                );
            }
            if candidates.len() >= limit {
                return;
            }
            continue;
        }
        let Some(alternative_head) = (0..marker)
            .rev()
            .filter(|head| !incoming.contains(head))
            .find(|head| is_predicate(*head, &primary.links))
        else {
            continue;
        };
        let mut links = primary.links.clone();
        links.retain(|link| !(link.right == marker && link.kind == SyntacticLinkKind::Complement));
        push_variant_link(
            &mut links,
            alternative_head,
            marker,
            SyntacticLinkKind::Complement,
            0.49,
        );
        add_candidate(
            candidates,
            primary,
            links,
            GrammarParseVariant::ComplementAttachment {
                marker,
                head: alternative_head,
            },
            0.06,
            backend,
            backend_parse_index,
            limit,
        );
        if candidates.len() >= limit {
            return;
        }
    }
}

fn add_phrasal_particle_variants(
    candidates: &mut Vec<RankedGrammarParse>,
    primary: &RankedGrammarParse,
    words: &[String],
    profile: GrammarRuleSet,
    backend: GrammarBackend,
    backend_parse_index: usize,
    limit: usize,
) {
    for (particle, word) in words.iter().enumerate().skip(1) {
        if !profile.phrasal_particles.contains(&word.as_str()) {
            continue;
        }
        let modifier = primary
            .links
            .iter()
            .find(|link| link.right == particle && link.kind == SyntacticLinkKind::Modifier);
        let preposition = primary
            .links
            .iter()
            .find(|link| link.left == particle && link.kind == SyntacticLinkKind::Preposition);
        let (Some(modifier), Some(preposition)) = (modifier, preposition) else {
            continue;
        };

        let mut particle_links = primary.links.clone();
        particle_links
            .retain(|link| !(link.left == particle && link.kind == SyntacticLinkKind::Preposition));
        push_variant_link(
            &mut particle_links,
            modifier.left,
            preposition.right,
            SyntacticLinkKind::Object,
            0.55,
        );
        add_candidate(
            candidates,
            primary,
            particle_links,
            GrammarParseVariant::PhrasalParticle {
                particle,
                as_particle: true,
            },
            0.04,
            backend,
            backend_parse_index,
            limit,
        );

        let mut preposition_links = primary.links.clone();
        preposition_links
            .retain(|link| !(link.right == particle && link.kind == SyntacticLinkKind::Modifier));
        add_candidate(
            candidates,
            primary,
            preposition_links,
            GrammarParseVariant::PhrasalParticle {
                particle,
                as_particle: false,
            },
            0.05,
            backend,
            backend_parse_index,
            limit,
        );
        if candidates.len() >= limit {
            return;
        }
    }
}

fn add_relative_clause_variants(
    candidates: &mut Vec<RankedGrammarParse>,
    primary: &RankedGrammarParse,
    backend: GrammarBackend,
    backend_parse_index: usize,
    limit: usize,
) {
    for apposition in primary
        .links
        .iter()
        .filter(|link| link.kind == SyntacticLinkKind::Apposition)
    {
        let marker = apposition.right;
        if !primary.links.iter().any(|link| {
            link.left == marker
                && matches!(
                    link.kind,
                    SyntacticLinkKind::Complement | SyntacticLinkKind::Subject
                )
        }) {
            continue;
        }
        let Some(head) = (0..apposition.left)
            .rev()
            .find(|index| is_nominal(*index, &primary.links))
        else {
            continue;
        };
        let mut links = primary.links.clone();
        links.retain(|link| !(link.right == marker && link.kind == SyntacticLinkKind::Apposition));
        push_variant_link(
            &mut links,
            head,
            marker,
            SyntacticLinkKind::Apposition,
            0.47,
        );
        add_candidate(
            candidates,
            primary,
            links,
            GrammarParseVariant::RelativeClause { marker, head },
            0.07,
            backend,
            backend_parse_index,
            limit,
        );
        if candidates.len() >= limit {
            return;
        }
    }
}

fn add_punctuation_island_variants(
    candidates: &mut Vec<RankedGrammarParse>,
    primary: &RankedGrammarParse,
    words: &[String],
    profile: GrammarRuleSet,
    backend: GrammarBackend,
    backend_parse_index: usize,
    limit: usize,
) {
    for (marker, word) in words
        .iter()
        .enumerate()
        .take(words.len().saturating_sub(1))
        .skip(1)
    {
        if !profile.parenthetical_markers.contains(&word.as_str()) {
            continue;
        }
        for attach_left in [true, false] {
            let anchor = if attach_left { marker - 1 } else { marker + 1 };
            let mut links = primary.links.clone();
            links.retain(|link| {
                link.kind != SyntacticLinkKind::Parenthetical
                    || (link.left != marker && link.right != marker)
            });
            push_variant_link(
                &mut links,
                anchor.min(marker),
                anchor.max(marker),
                SyntacticLinkKind::Parenthetical,
                0.48,
            );
            add_candidate(
                candidates,
                primary,
                links,
                GrammarParseVariant::PunctuationIsland {
                    marker,
                    attach_left,
                },
                if attach_left { 0.05 } else { 0.06 },
                backend,
                backend_parse_index,
                limit,
            );
        }
        if candidates.len() >= limit {
            return;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn add_candidate(
    candidates: &mut Vec<RankedGrammarParse>,
    primary: &RankedGrammarParse,
    mut links: Vec<SyntacticLink>,
    variant: GrammarParseVariant,
    penalty: f32,
    backend: GrammarBackend,
    backend_parse_index: usize,
    limit: usize,
) {
    if candidates.len() >= limit {
        return;
    }
    canonicalize_links(&mut links);
    if candidates
        .iter()
        .any(|candidate| same_link_graph(&candidate.links, &links))
    {
        return;
    }
    candidates.push(RankedGrammarParse {
        id: parse_id(backend, backend_parse_index, &variant),
        confidence: normalized_confidence(&links),
        rank: (primary.rank - penalty).clamp(0.0, 1.0),
        status: primary.status,
        provenance: GrammarParseProvenance {
            backend,
            backend_parse_index,
            variant,
        },
        links,
    });
}

fn coordination_indices(
    words: &[String],
    links: &[SyntacticLink],
    profile: Option<GrammarRuleSet>,
) -> BTreeSet<WordIndex> {
    let mut indices = BTreeSet::new();
    if let Some(profile) = profile {
        for (index, word) in words.iter().enumerate() {
            if profile.conjunctions.contains(&word.as_str()) {
                indices.insert(index);
            }
        }
    }
    for link in links {
        if link.kind == SyntacticLinkKind::Coordination && link.right >= link.left + 2 {
            indices.insert(link.left + 1);
        }
    }
    indices
}

fn is_predicate(index: WordIndex, links: &[SyntacticLink]) -> bool {
    links.iter().any(|link| {
        (link.left == index
            && matches!(
                link.kind,
                SyntacticLinkKind::Object | SyntacticLinkKind::Complement
            ))
            || (link.right == index
                && matches!(
                    link.kind,
                    SyntacticLinkKind::Subject
                        | SyntacticLinkKind::Auxiliary
                        | SyntacticLinkKind::InfinitivalMarker
                ))
    })
}

fn is_nominal(index: WordIndex, links: &[SyntacticLink]) -> bool {
    links.iter().any(|link| {
        (link.right == index
            && matches!(
                link.kind,
                SyntacticLinkKind::Determiner
                    | SyntacticLinkKind::Object
                    | SyntacticLinkKind::Preposition
                    | SyntacticLinkKind::Modifier
            ))
            || (link.left == index && link.kind == SyntacticLinkKind::Subject)
    })
}

fn push_variant_link(
    links: &mut Vec<SyntacticLink>,
    left: WordIndex,
    right: WordIndex,
    kind: SyntacticLinkKind,
    confidence: f32,
) {
    if left == right
        || links
            .iter()
            .any(|link| link.left == left && link.right == right && link.kind == kind)
    {
        return;
    }
    links.push(SyntacticLink {
        left: left.min(right),
        right: left.max(right),
        kind,
        confidence,
        source: SyntacticLinkSource::AmbiguityVariant,
    });
}

fn normalized_confidence(links: &[SyntacticLink]) -> f32 {
    if links.is_empty() {
        return 0.0;
    }
    (links.iter().map(|link| link.confidence).sum::<f32>() / links.len() as f32).clamp(0.0, 1.0)
}

fn normalized_rank(links: &[SyntacticLink], word_count: usize, status: GrammarParseStatus) -> f32 {
    if word_count == 0 {
        return 0.0;
    }
    let linked = links
        .iter()
        .flat_map(|link| [link.left, link.right])
        .filter(|index| *index < word_count)
        .collect::<BTreeSet<_>>()
        .len();
    let coverage = linked as f32 / word_count as f32;
    let completeness = if status == GrammarParseStatus::Complete {
        1.0
    } else {
        0.65
    };
    (normalized_confidence(links) * 0.65 + coverage * 0.25 + completeness * 0.10).clamp(0.0, 1.0)
}

fn canonicalize_links(links: &mut Vec<SyntacticLink>) {
    links.sort_by(|left, right| {
        left.left
            .cmp(&right.left)
            .then(left.right.cmp(&right.right))
            .then((left.kind as u8).cmp(&(right.kind as u8)))
            .then_with(|| right.confidence.total_cmp(&left.confidence))
            .then((left.source as u8).cmp(&(right.source as u8)))
    });
    links.dedup_by(|left, right| {
        left.left == right.left && left.right == right.right && left.kind == right.kind
    });
}

fn same_link_graph(left: &[SyntacticLink], right: &[SyntacticLink]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.left == right.left && left.right == right.right && left.kind == right.kind
        })
}

fn compare_ranked_parses(left: &RankedGrammarParse, right: &RankedGrammarParse) -> Ordering {
    right
        .rank
        .total_cmp(&left.rank)
        .then_with(|| right.confidence.total_cmp(&left.confidence))
        .then_with(|| left.id.cmp(&right.id))
}

fn parse_id(
    backend: GrammarBackend,
    backend_parse_index: usize,
    variant: &GrammarParseVariant,
) -> GrammarParseId {
    let backend = match backend {
        GrammarBackend::TonguesRules => "tongues-rules",
        GrammarBackend::UdPipe => "ud-pipe",
    };
    let variant = match variant {
        GrammarParseVariant::BackendPrimary => "primary".to_string(),
        GrammarParseVariant::PrepositionalAttachment {
            preposition,
            object,
            head,
        } => format!("pp-{preposition}-{object}-head-{head}"),
        GrammarParseVariant::CoordinationScope {
            conjunction,
            left,
            right,
        } => format!("coord-{conjunction}-{left}-{right}"),
        GrammarParseVariant::PartOfSpeech { word, pos } => {
            format!("pos-{word}-{}", pos_label(*pos))
        }
        GrammarParseVariant::ComplementAttachment { marker, head } => {
            format!("complement-{marker}-head-{head}")
        }
        GrammarParseVariant::PhrasalParticle {
            particle,
            as_particle,
        } => format!(
            "particle-{particle}-{}",
            if *as_particle {
                "particle"
            } else {
                "adposition"
            }
        ),
        GrammarParseVariant::RelativeClause { marker, head } => {
            format!("relative-{marker}-head-{head}")
        }
        GrammarParseVariant::PunctuationIsland {
            marker,
            attach_left,
        } => format!(
            "island-{marker}-{}",
            if *attach_left { "left" } else { "right" }
        ),
    };
    GrammarParseId(format!("grammar:{backend}:{backend_parse_index}:{variant}"))
}

fn pos_label(pos: crate::syntax::PartOfSpeech) -> &'static str {
    match pos {
        crate::syntax::PartOfSpeech::Noun => "noun",
        crate::syntax::PartOfSpeech::Verb => "verb",
        crate::syntax::PartOfSpeech::Auxiliary => "auxiliary",
        crate::syntax::PartOfSpeech::Determiner => "determiner",
        crate::syntax::PartOfSpeech::Preposition => "preposition",
        crate::syntax::PartOfSpeech::Pronoun => "pronoun",
        crate::syntax::PartOfSpeech::Adverb => "adverb",
        crate::syntax::PartOfSpeech::Adjective => "adjective",
        crate::syntax::PartOfSpeech::Conjunction => "conjunction",
        crate::syntax::PartOfSpeech::Particle => "particle",
        crate::syntax::PartOfSpeech::ProperName => "proper-name",
        crate::syntax::PartOfSpeech::Unknown => "unknown",
    }
}
