use crate::segment::TerminalPunctuation;
use crate::syntax::{
    LinkParserCommandBackend, PartOfSpeech, ProsodicRole, SentenceSyntaxAnalysis, SyntacticLink,
    SyntacticLinkKind, SyntacticLinkParse, SyntaxToken, link, normalize_syntax_word, push_link,
    use_link_parser_command_backend,
};

pub fn parse_link_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
) -> SentenceSyntaxAnalysis {
    if use_link_parser_command_backend() {
        if let Some(mut analysis) = LinkParserCommandBackend::new().parse(words, terminal) {
            annotate_tokens(&mut analysis);
            return analysis;
        }
    }
    parse_heuristic_link_grammar(words, terminal)
}

pub fn parse_heuristic_link_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
) -> SentenceSyntaxAnalysis {
    let pairs = words
        .iter()
        .filter_map(|word| {
            let normalized = normalize_syntax_word(word);
            (!normalized.is_empty()).then(|| (word.clone(), normalized))
        })
        .collect::<Vec<_>>();
    let normalized = pairs
        .iter()
        .map(|(_, normalized)| normalized.clone())
        .collect::<Vec<_>>();
    let links = build_links(&normalized);
    let parse = SyntacticLinkParse { links, rank: 1.0 };
    let tokens = normalized
        .iter()
        .enumerate()
        .map(|(word_index, word)| {
            let mut syntactic_links = parse
                .links
                .iter()
                .filter_map(|link| {
                    (link.left == word_index || link.right == word_index).then_some(link.kind)
                })
                .collect::<Vec<_>>();
            syntactic_links.sort_unstable_by_key(|kind| *kind as u8);
            syntactic_links.dedup();
            SyntaxToken {
                word_index,
                text: pairs[word_index].0.clone(),
                pos: disambiguate_pos_from_links(word_index, base_pos(word), &parse.links),
                prosodic_role: prosodic_role_for_word(word, &syntactic_links),
                syntactic_links,
            }
        })
        .collect();

    SentenceSyntaxAnalysis {
        tokens,
        link_parses: vec![parse],
        raw_link_grammar_parses: Vec::new(),
        terminal,
    }
}

fn build_links(words: &[String]) -> Vec<SyntacticLink> {
    let mut links = Vec::new();
    for (index, window) in words.windows(2).enumerate() {
        let left = window[0].as_str();
        let right = window[1].as_str();
        if left == "to" && is_likely_verb(right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::InfinitivalMarker, 0.92),
            );
        }
        if is_determiner(left) && is_likely_nominal(right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Determiner, 0.83),
            );
        }
        if is_auxiliary(left) && is_likely_verb(right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Auxiliary, 0.82),
            );
        }
        if is_preposition(left) && is_likely_nominal(right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Preposition, 0.8),
            );
        }
        if is_nominal_modifier(left, right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::NounCompound, 0.73),
            );
        }
        if is_modifier_pair(left, right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Modifier, 0.72),
            );
        }
        if is_appositive_pair(left, right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Apposition, 0.7),
            );
        }
        if is_vocative_opener(left) && is_likely_nominal(right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Vocative, 0.82),
            );
        }
        if is_parenthetical_marker(left) || is_parenthetical_marker(right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Parenthetical, 0.58),
            );
        }
    }

    push_prepositional_phrase_links(words, &mut links);
    push_possessive_links(words, &mut links);
    push_modifier_phrase_links(words, &mut links);
    push_auxiliary_phrase_links(words, &mut links);
    push_core_clause_links(words, &mut links);
    push_complement_links(words, &mut links);
    push_fronted_clause_marker_links(words, &mut links);
    push_relative_clause_links(words, &mut links);
    push_particle_links(words, &mut links);
    push_passive_participle_links(words, &mut links);
    push_coordination_links(words, &mut links);
    push_contrast_links(words, &mut links);
    links
}

fn push_prepositional_phrase_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for preposition_index in 0..words.len() {
        if !is_preposition(&words[preposition_index]) {
            continue;
        }
        if let Some(object_index) = words
            .iter()
            .enumerate()
            .skip(preposition_index + 1)
            .take(4)
            .find_map(|(index, word)| {
                (is_likely_nominal(word) && !is_modifier_only(word)).then_some(index)
            })
        {
            push_link(
                links,
                link(
                    preposition_index,
                    object_index,
                    SyntacticLinkKind::Preposition,
                    0.8,
                ),
            );
        }
    }
}

fn push_possessive_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for possessive_index in 0..words.len().saturating_sub(1) {
        if !is_possessive_nominal(&words[possessive_index]) {
            continue;
        }
        if let Some(head_index) = words
            .iter()
            .enumerate()
            .skip(possessive_index + 1)
            .take(4)
            .find_map(|(index, word)| {
                (is_likely_nominal(word) && !is_modifier_only(word)).then_some(index)
            })
        {
            push_link(
                links,
                link(
                    possessive_index,
                    head_index,
                    SyntacticLinkKind::Determiner,
                    0.81,
                ),
            );
        }
    }
}

fn push_modifier_phrase_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for modifier_index in 0..words.len() {
        if !is_adverb(&words[modifier_index]) {
            continue;
        }
        if let Some(head_index) = (0..modifier_index)
            .rev()
            .take(5)
            .find(|index| is_likely_verb(&words[*index]) || is_adjective(&words[*index]))
        {
            push_link(
                links,
                link(
                    head_index,
                    modifier_index,
                    SyntacticLinkKind::Modifier,
                    0.68,
                ),
            );
        }
    }
}

fn push_auxiliary_phrase_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for auxiliary_index in 0..words.len() {
        if !is_auxiliary(&words[auxiliary_index]) {
            continue;
        }
        if let Some(verb_index) = words
            .iter()
            .enumerate()
            .skip(auxiliary_index + 1)
            .take(4)
            .find_map(|(index, word)| is_likely_verb(word).then_some(index))
        {
            push_link(
                links,
                link(
                    auxiliary_index,
                    verb_index,
                    SyntacticLinkKind::Auxiliary,
                    0.82,
                ),
            );
        }
    }
}

fn push_core_clause_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for predicate_index in 0..words.len() {
        if !(is_likely_verb(&words[predicate_index]) || is_auxiliary(&words[predicate_index])) {
            continue;
        }
        if let Some(subject_index) = (0..predicate_index)
            .rev()
            .find(|index| is_subject_candidate(&words[*index]))
        {
            push_link(
                links,
                link(
                    subject_index,
                    predicate_index,
                    SyntacticLinkKind::Subject,
                    0.8,
                ),
            );
        }
        if is_copula(&words[predicate_index]) {
            continue;
        }
        if let Some(object_index) = words
            .iter()
            .enumerate()
            .skip(predicate_index + 1)
            .take(5)
            .take_while(|(_, word)| !is_clause_marker(word) && !is_coordination_conjunction(word))
            .find_map(|(index, word)| {
                (is_likely_nominal(word) && !is_modifier_only(word)).then_some(index)
            })
        {
            push_link(
                links,
                link(
                    predicate_index,
                    object_index,
                    SyntacticLinkKind::Object,
                    0.78,
                ),
            );
        }
    }
}

fn push_complement_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for predicate_index in 0..words.len() {
        let word = words[predicate_index].as_str();
        if is_copula(word) {
            if let Some(complement_index) = words
                .iter()
                .enumerate()
                .skip(predicate_index + 1)
                .take(5)
                .find_map(|(index, word)| {
                    (is_likely_nominal(word) || is_adjective(word)).then_some(index)
                })
            {
                push_link(
                    links,
                    link(
                        predicate_index,
                        complement_index,
                        SyntacticLinkKind::Complement,
                        0.76,
                    ),
                );
            }
        }

        if !is_likely_verb(word) && !is_auxiliary(word) {
            continue;
        }
        if let Some(complement_index) = words
            .iter()
            .enumerate()
            .skip(predicate_index + 1)
            .take(6)
            .find_map(|(index, word)| is_clause_marker(word).then_some(index))
        {
            push_link(
                links,
                link(
                    predicate_index,
                    complement_index,
                    SyntacticLinkKind::Complement,
                    0.69,
                ),
            );
        }
    }
}

fn push_fronted_clause_marker_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for marker_index in 0..words.len() {
        if !is_clause_marker(&words[marker_index]) {
            continue;
        }
        if let Some(predicate_index) = words
            .iter()
            .enumerate()
            .skip(marker_index + 1)
            .take(6)
            .find_map(|(index, word)| (is_likely_verb(word) || is_auxiliary(word)).then_some(index))
        {
            push_link(
                links,
                link(
                    marker_index,
                    predicate_index,
                    SyntacticLinkKind::Complement,
                    0.66,
                ),
            );
        }
    }
}

fn push_relative_clause_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for marker_index in 1..words.len() {
        if !is_relative_marker(&words[marker_index]) {
            continue;
        }
        let Some(head_index) = (0..marker_index)
            .rev()
            .take(4)
            .find(|index| is_likely_nominal(&words[*index]) && !is_modifier_only(&words[*index]))
        else {
            continue;
        };
        if let Some(predicate_index) = words
            .iter()
            .enumerate()
            .skip(marker_index + 1)
            .take(6)
            .find_map(|(index, word)| (is_likely_verb(word) || is_auxiliary(word)).then_some(index))
        {
            push_link(
                links,
                link(
                    head_index,
                    marker_index,
                    SyntacticLinkKind::Apposition,
                    0.62,
                ),
            );
            push_link(
                links,
                link(
                    marker_index,
                    predicate_index,
                    SyntacticLinkKind::Complement,
                    0.65,
                ),
            );
        }
    }
}

fn push_particle_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for particle_index in 1..words.len() {
        if !is_phrasal_particle(&words[particle_index]) {
            continue;
        }
        if let Some(verb_index) = (0..particle_index)
            .rev()
            .take(2)
            .find(|index| is_likely_verb(&words[*index]))
        {
            push_link(
                links,
                link(
                    verb_index,
                    particle_index,
                    SyntacticLinkKind::Modifier,
                    0.66,
                ),
            );
        }
    }
}

fn push_passive_participle_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for participle_index in 1..words.len() {
        if !is_past_participle(&words[participle_index]) {
            continue;
        }
        let Some(auxiliary_index) = (0..participle_index).rev().take(3).find(|index| {
            is_copula(&words[*index]) || words[*index] == "get" || words[*index] == "got"
        }) else {
            continue;
        };
        if let Some(subject_index) = (0..auxiliary_index)
            .rev()
            .take(5)
            .find(|index| is_subject_candidate(&words[*index]))
        {
            push_link(
                links,
                link(
                    subject_index,
                    participle_index,
                    SyntacticLinkKind::Subject,
                    0.64,
                ),
            );
        }
    }
}

fn push_coordination_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for conjunction_index in 1..words.len().saturating_sub(1) {
        if !is_coordination_conjunction(&words[conjunction_index]) {
            continue;
        }
        push_link(
            links,
            link(
                conjunction_index - 1,
                conjunction_index + 1,
                SyntacticLinkKind::Coordination,
                0.74,
            ),
        );
        push_link(
            links,
            link(
                conjunction_index,
                conjunction_index + 1,
                SyntacticLinkKind::Coordination,
                0.74,
            ),
        );
    }
}

fn push_contrast_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for (not_index, word) in words.iter().enumerate() {
        if !is_contrast_negator(word) {
            continue;
        }
        if let Some(but_index) = words
            .iter()
            .enumerate()
            .skip(not_index + 1)
            .find_map(|(index, word)| (word == "but").then_some(index))
        {
            push_link(
                links,
                link(not_index, but_index, SyntacticLinkKind::ContrastPair, 0.91),
            );
        }
    }
}

fn disambiguate_pos_from_links(
    word_index: usize,
    base: PartOfSpeech,
    links: &[SyntacticLink],
) -> PartOfSpeech {
    let has_incoming = |kind| {
        links
            .iter()
            .any(|link| link.right == word_index && link.kind == kind)
    };
    match base {
        PartOfSpeech::Noun if has_incoming(SyntacticLinkKind::Auxiliary) => PartOfSpeech::Verb,
        PartOfSpeech::Verb if has_incoming(SyntacticLinkKind::Determiner) => PartOfSpeech::Noun,
        _ => base,
    }
}

fn annotate_tokens(analysis: &mut SentenceSyntaxAnalysis) {
    let links = analysis
        .primary_parse()
        .map(|parse| parse.links.clone())
        .unwrap_or_default();
    for token in &mut analysis.tokens {
        let normalized = normalize_syntax_word(&token.text);
        let mut syntactic_links = links
            .iter()
            .filter_map(|link| {
                (link.left == token.word_index || link.right == token.word_index)
                    .then_some(link.kind)
            })
            .collect::<Vec<_>>();
        syntactic_links.sort_unstable_by_key(|kind| *kind as u8);
        syntactic_links.dedup();
        token.pos = disambiguate_pos_from_links(token.word_index, base_pos(&normalized), &links);
        token.prosodic_role = prosodic_role_for_word(&normalized, &syntactic_links);
        token.syntactic_links = syntactic_links;
    }
}

fn prosodic_role_for_word(word: &str, links: &[SyntacticLinkKind]) -> ProsodicRole {
    if links.contains(&SyntacticLinkKind::ContrastPair) {
        ProsodicRole::Contrastive
    } else if is_function_word(word) {
        ProsodicRole::FunctionWeak
    } else if links.contains(&SyntacticLinkKind::Object)
        || links.contains(&SyntacticLinkKind::Complement)
    {
        ProsodicRole::Focus
    } else {
        ProsodicRole::Content
    }
}

fn is_clause_marker(word: &str) -> bool {
    is_complementizer(word) || is_subordinating_conjunction(word)
}

fn is_relative_marker(word: &str) -> bool {
    matches!(word, "that" | "which" | "who" | "whom" | "whose")
}

fn is_subject_candidate(word: &str) -> bool {
    is_likely_nominal(word)
        && !is_preposition(word)
        && (!is_modifier_only(word) || is_demonstrative_pronoun(word))
}

fn is_modifier_pair(left: &str, right: &str) -> bool {
    (is_adjective(left) && is_likely_nominal(right))
        || (is_adverb(left) && (is_adjective(right) || is_likely_verb(right)))
}

fn is_modifier_only(word: &str) -> bool {
    is_adjective(word) || is_adverb(word) || is_determiner(word)
}

fn is_nominal_modifier(left: &str, right: &str) -> bool {
    is_likely_nominal(left)
        && is_likely_nominal(right)
        && !is_modifier_only(left)
        && !is_modifier_only(right)
        && !is_proper_name(left)
        && !is_pronoun(left)
        && !is_pronoun(right)
        && !is_likely_verb(left)
        && !is_likely_verb(right)
}

fn is_appositive_pair(left: &str, right: &str) -> bool {
    (is_common_appositive_head(left) && is_proper_name(right))
        || (is_proper_name(left) && is_common_appositive_head(right))
}

fn is_possessive_nominal(word: &str) -> bool {
    word.strip_suffix("'s")
        .is_some_and(|stem| !stem.is_empty() && is_likely_nominal(stem))
}

fn is_phrasal_particle(word: &str) -> bool {
    matches!(
        word,
        "about"
            | "away"
            | "back"
            | "down"
            | "in"
            | "off"
            | "on"
            | "out"
            | "over"
            | "through"
            | "up"
    )
}

fn is_past_participle(word: &str) -> bool {
    word.ends_with("ed")
        || matches!(
            word,
            "bought"
                | "chosen"
                | "done"
                | "given"
                | "gone"
                | "known"
                | "left"
                | "made"
                | "read"
                | "seen"
                | "taken"
                | "thought"
                | "thrown"
                | "told"
                | "written"
        )
}

pub const AUXILIARIES: &[&str] = &[
    "am",
    "are",
    "aren't",
    "is",
    "isn't",
    "was",
    "wasn't",
    "were",
    "weren't",
    "do",
    "don't",
    "does",
    "doesn't",
    "did",
    "didn't",
    "have",
    "haven't",
    "has",
    "hasn't",
    "had",
    "hadn't",
    "can",
    "can't",
    "could",
    "couldn't",
    "will",
    "won't",
    "would",
    "wouldn't",
    "shall",
    "should",
    "shouldn't",
    "may",
    "might",
    "must",
    "ought",
    "need",
    "dare",
    "be",
    "been",
    "being",
];

pub const COPULAS: &[&str] = &[
    "am", "are", "aren't", "is", "isn't", "was", "wasn't", "were", "weren't", "be", "been", "being",
];

pub const DETERMINERS: &[&str] = &[
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "our", "their", "his", "her",
    "its", "all", "another", "any", "both", "each", "either", "every", "many", "much", "no",
    "some", "such", "what", "which",
];

pub const PREPOSITIONS: &[&str] = &[
    "about",
    "above",
    "across",
    "after",
    "against",
    "along",
    "around",
    "at",
    "before",
    "behind",
    "below",
    "beside",
    "besides",
    "between",
    "by",
    "during",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "like",
    "near",
    "of",
    "off",
    "on",
    "onto",
    "out",
    "over",
    "through",
    "throughout",
    "to",
    "under",
    "until",
    "up",
    "with",
    "without",
];

pub const COORDINATION_CONJUNCTIONS: &[&str] = &["and", "or", "but", "nor"];

pub const SUBORDINATING_CONJUNCTIONS: &[&str] = &[
    "after", "although", "as", "because", "before", "if", "since", "though", "unless", "until",
    "when", "where", "whether", "while",
];

pub const COMPLEMENTIZERS: &[&str] = &["that", "whether", "if", "who", "what", "which", "how"];

pub const PRONOUNS: &[&str] = &[
    "i", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them", "who", "whom",
    "what", "which",
];

pub const NOMINAL_FUNCTION_WORDS: &[&str] = &[
    "i", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them", "who", "what",
    "which", "this", "that", "these", "those",
];

pub const DEMONSTRATIVE_PRONOUNS: &[&str] = &["this", "that", "these", "those"];

pub const LIKELY_VERBS: &[&str] = &[
    "act",
    "appear",
    "arrive",
    "ask",
    "asked",
    "be",
    "believe",
    "bought",
    "buy",
    "came",
    "chase",
    "chased",
    "choose",
    "close",
    "coming",
    "come",
    "comply",
    "conduct",
    "console",
    "contrast",
    "contrasted",
    "decide",
    "did",
    "die",
    "do",
    "eat",
    "fix",
    "gave",
    "give",
    "go",
    "goes",
    "going",
    "had",
    "has",
    "have",
    "hear",
    "help",
    "hit",
    "hope",
    "inhale",
    "inspect",
    "invite",
    "know",
    "knows",
    "lead",
    "left",
    "like",
    "likes",
    "made",
    "make",
    "meet",
    "met",
    "object",
    "operate",
    "parse",
    "permit",
    "present",
    "produce",
    "project",
    "put",
    "ran",
    "read",
    "realize",
    "rebel",
    "record",
    "remember",
    "result",
    "refuse",
    "rose",
    "run",
    "runs",
    "said",
    "saw",
    "say",
    "see",
    "seem",
    "seems",
    "seen",
    "smiled",
    "subject",
    "talk",
    "tell",
    "think",
    "thinks",
    "thought",
    "throw",
    "thrown",
    "told",
    "took",
    "turn",
    "turned",
    "use",
    "walk",
    "walked",
    "want",
    "wanted",
    "wants",
    "went",
    "win",
    "wind",
    "work",
    "works",
    "write",
    "wrote",
];

pub const VERB_SUFFIXES: &[&str] = &["ed", "ing"];

pub const ADJECTIVES: &[&str] = &[
    "administrative",
    "afraid",
    "angry",
    "beautiful",
    "big",
    "black",
    "bright",
    "careful",
    "certain",
    "clear",
    "dark",
    "easy",
    "excellent",
    "expensive",
    "fast",
    "female",
    "fortunate",
    "good",
    "great",
    "grotesque",
    "happy",
    "heavy",
    "important",
    "impatient",
    "inexpensive",
    "large",
    "likely",
    "long",
    "lyrical",
    "medical",
    "necessary",
    "new",
    "obvious",
    "old",
    "patient",
    "possible",
    "ready",
    "relaxed",
    "rude",
    "short",
    "slow",
    "small",
    "stupid",
    "sure",
    "tired",
    "ugly",
    "unfortunate",
    "valid",
    "white",
];

pub const ADJECTIVE_SUFFIXES: &[&str] = &["able", "al", "ful", "ic", "ical", "ive", "less", "ous"];

pub const ADVERBS: &[&str] = &[
    "already",
    "apparently",
    "broadly",
    "delicately",
    "eventually",
    "fortunately",
    "generally",
    "gradually",
    "initially",
    "just",
    "mainly",
    "never",
    "not",
    "now",
    "often",
    "particularly",
    "presumably",
    "quickly",
    "really",
    "recently",
    "sadly",
    "sometimes",
    "soon",
    "specifically",
    "straight",
    "ultimately",
    "usually",
    "very",
];

pub const ADVERB_SUFFIXES: &[&str] = &["ly"];

pub const COMMON_APPOSITIVE_HEADS: &[&str] = &[
    "actress",
    "author",
    "brother",
    "cousin",
    "doctor",
    "expert",
    "friend",
    "man",
    "mother",
    "president",
    "singer",
    "sister",
    "student",
    "uncle",
    "woman",
];

pub const PROPER_NAMES: &[&str] = &[
    "abrams", "alfred", "alice", "ann", "anne", "baird", "bob", "charles", "chris", "clinton",
    "david", "dick", "einstein", "emily", "fred", "grace", "janet", "joan", "joe", "john", "ken",
    "mary", "michael", "nixon", "oj", "rod", "ruth", "sally", "smith", "stuart", "ted", "thomas",
    "whoopi",
];

pub const VOCATIVE_OPENERS: &[&str] = &["hey", "oh"];

pub const PARENTHETICAL_MARKERS: &[&str] = &[
    "apparently",
    "fortunately",
    "however",
    "particularly",
    "presumably",
    "therefore",
];

pub const CONTRAST_NEGATORS: &[&str] = &["not", "n't"];

pub fn base_pos(word: &str) -> PartOfSpeech {
    if is_auxiliary(word) {
        PartOfSpeech::Auxiliary
    } else if is_determiner(word) {
        PartOfSpeech::Determiner
    } else if is_preposition(word) {
        PartOfSpeech::Preposition
    } else if is_coordination_conjunction(word) || is_subordinating_conjunction(word) {
        PartOfSpeech::Conjunction
    } else if is_adverb(word) {
        PartOfSpeech::Adverb
    } else if is_adjective(word) {
        PartOfSpeech::Adjective
    } else if is_vocative_opener(word) {
        PartOfSpeech::Particle
    } else if is_proper_name(word) {
        PartOfSpeech::ProperName
    } else if is_pronoun(word) {
        PartOfSpeech::Pronoun
    } else if is_likely_verb(word) {
        PartOfSpeech::Verb
    } else {
        PartOfSpeech::Noun
    }
}

pub fn is_function_word(word: &str) -> bool {
    is_auxiliary(word)
        || is_determiner(word)
        || is_preposition(word)
        || is_coordination_conjunction(word)
        || is_subordinating_conjunction(word)
        || is_complementizer(word)
}

pub fn is_auxiliary(word: &str) -> bool {
    AUXILIARIES.contains(&word)
}

pub fn is_copula(word: &str) -> bool {
    COPULAS.contains(&word)
}

pub fn is_determiner(word: &str) -> bool {
    DETERMINERS.contains(&word)
}

pub fn is_preposition(word: &str) -> bool {
    PREPOSITIONS.contains(&word)
}

pub fn is_coordination_conjunction(word: &str) -> bool {
    COORDINATION_CONJUNCTIONS.contains(&word)
}

pub fn is_subordinating_conjunction(word: &str) -> bool {
    SUBORDINATING_CONJUNCTIONS.contains(&word)
}

pub fn is_complementizer(word: &str) -> bool {
    COMPLEMENTIZERS.contains(&word)
}

pub fn is_likely_nominal(word: &str) -> bool {
    !is_function_word(word) || NOMINAL_FUNCTION_WORDS.contains(&word)
}

pub fn is_likely_verb(word: &str) -> bool {
    LIKELY_VERBS.contains(&word) || VERB_SUFFIXES.iter().any(|suffix| word.ends_with(suffix))
}

pub fn is_adjective(word: &str) -> bool {
    ADJECTIVES.contains(&word)
        || ADJECTIVE_SUFFIXES
            .iter()
            .any(|suffix| word.ends_with(suffix))
}

pub fn is_adverb(word: &str) -> bool {
    ADVERBS.contains(&word) || ADVERB_SUFFIXES.iter().any(|suffix| word.ends_with(suffix))
}

pub fn is_common_appositive_head(word: &str) -> bool {
    COMMON_APPOSITIVE_HEADS.contains(&word)
}

pub fn is_proper_name(word: &str) -> bool {
    PROPER_NAMES.contains(&word)
}

pub fn is_pronoun(word: &str) -> bool {
    PRONOUNS.contains(&word)
}

pub fn is_demonstrative_pronoun(word: &str) -> bool {
    DEMONSTRATIVE_PRONOUNS.contains(&word)
}

pub fn is_vocative_opener(word: &str) -> bool {
    VOCATIVE_OPENERS.contains(&word)
}

pub fn is_parenthetical_marker(word: &str) -> bool {
    PARENTHETICAL_MARKERS.contains(&word)
}

pub fn is_contrast_negator(word: &str) -> bool {
    CONTRAST_NEGATORS.contains(&word)
}
