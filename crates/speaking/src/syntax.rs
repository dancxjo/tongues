use serde::{Deserialize, Serialize};

use crate::segment::TerminalPunctuation;

pub type WordIndex = usize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SentenceSyntaxAnalysis {
    pub tokens: Vec<SyntaxToken>,
    pub link_parses: Vec<SyntacticLinkParse>,
    pub terminal: Option<TerminalPunctuation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntaxToken {
    pub word_index: WordIndex,
    pub text: String,
    pub pos: PartOfSpeech,
    pub prosodic_role: ProsodicRole,
    pub syntactic_links: Vec<SyntacticLinkKind>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntacticLinkParse {
    pub links: Vec<SyntacticLink>,
    pub rank: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntacticLink {
    pub left: WordIndex,
    pub right: WordIndex,
    pub kind: SyntacticLinkKind,
    pub confidence: f32,
    pub source: SyntacticLinkSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SyntacticLinkKind {
    Subject,
    Object,
    Complement,
    InfinitivalMarker,
    Modifier,
    Determiner,
    Auxiliary,
    Preposition,
    Coordination,
    ContrastPair,
    NounCompound,
    Vocative,
    Apposition,
    Parenthetical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntacticLinkSource {
    HeuristicGrammarIsland,
    AmbiguityVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartOfSpeech {
    Noun,
    Verb,
    Auxiliary,
    Determiner,
    Preposition,
    Pronoun,
    Adverb,
    Adjective,
    Conjunction,
    Particle,
    ProperName,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProsodicRole {
    Content,
    FunctionWeak,
    FunctionStrong,
    Contrastive,
    Focus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentPattern {
    pub predicates: Vec<ContextPredicate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextPredicate {
    SyntacticLink(SyntacticLinkKind),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SyntaxRuleContext {
    pub word_links: Vec<WordSyntacticLinks>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordSyntacticLinks {
    pub word_index: WordIndex,
    pub links: Vec<SyntacticLinkKind>,
}

pub trait LinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicLinkGrammarParser;

impl LinkGrammarParser for HeuristicLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        parse_english_link_grammar(words, terminal)
    }
}

pub fn parse_english_link_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
) -> SentenceSyntaxAnalysis {
    let links = build_links(words);
    let parse = SyntacticLinkParse { links, rank: 1.0 };
    let tokens = words
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
                text: word.clone(),
                pos: disambiguate_pos_from_links(word_index, base_pos(word), &parse.links),
                prosodic_role: prosodic_role_for_word(word, &syntactic_links),
                syntactic_links,
            }
        })
        .collect();

    SentenceSyntaxAnalysis {
        tokens,
        link_parses: vec![parse],
        terminal,
    }
}

impl SentenceSyntaxAnalysis {
    pub fn primary_parse(&self) -> Option<&SyntacticLinkParse> {
        self.link_parses.first()
    }

    pub fn environment_patterns(&self) -> Vec<EnvironmentPattern> {
        self.link_parses
            .iter()
            .map(SyntacticLinkParse::as_environment_pattern)
            .collect()
    }

    pub fn rule_context(&self) -> SyntaxRuleContext {
        SyntaxRuleContext {
            word_links: self
                .tokens
                .iter()
                .map(|token| WordSyntacticLinks {
                    word_index: token.word_index,
                    links: token.syntactic_links.clone(),
                })
                .collect(),
        }
    }

    pub fn word_has_link(&self, word_index: WordIndex, kind: SyntacticLinkKind) -> bool {
        self.rule_context().word_has_link(word_index, kind)
    }

    pub fn matches_environment_pattern(&self, pattern: &EnvironmentPattern) -> bool {
        let Some(primary) = self.primary_parse() else {
            return false;
        };
        pattern.predicates.iter().all(|predicate| match predicate {
            ContextPredicate::SyntacticLink(kind) => {
                primary.links.iter().any(|link| link.kind == *kind)
            }
        })
    }
}

impl SyntacticLinkParse {
    pub fn as_environment_pattern(&self) -> EnvironmentPattern {
        let mut seen = std::collections::HashSet::new();
        let predicates = self
            .links
            .iter()
            .filter_map(|link| {
                seen.insert(link.kind)
                    .then_some(ContextPredicate::SyntacticLink(link.kind))
            })
            .collect();
        EnvironmentPattern { predicates }
    }
}

impl SyntaxRuleContext {
    pub fn word_has_link(&self, word_index: WordIndex, kind: SyntacticLinkKind) -> bool {
        self.word_links
            .iter()
            .find(|word| word.word_index == word_index)
            .is_some_and(|word| word.links.contains(&kind))
    }
}

fn build_links(words: &[String]) -> Vec<SyntacticLink> {
    let words = words
        .iter()
        .map(|word| normalize_syntax_word(word))
        .collect::<Vec<_>>();
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

    push_prepositional_phrase_links(&words, &mut links);
    push_modifier_phrase_links(&words, &mut links);
    push_auxiliary_phrase_links(&words, &mut links);
    push_core_clause_links(&words, &mut links);
    push_complement_links(&words, &mut links);
    push_coordination_links(&words, &mut links);
    push_contrast_links(&words, &mut links);
    links
}

fn normalize_syntax_word(word: &str) -> String {
    word.trim_matches(|character: char| !character.is_alphabetic() && character != '\'')
        .to_ascii_lowercase()
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
            .find_map(|(index, word)| is_complementizer(word).then_some(index))
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
        if !matches!(word.as_str(), "not" | "n't") {
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

fn link(left: usize, right: usize, kind: SyntacticLinkKind, confidence: f32) -> SyntacticLink {
    SyntacticLink {
        left,
        right,
        kind,
        confidence,
        source: SyntacticLinkSource::HeuristicGrammarIsland,
    }
}

fn push_link(links: &mut Vec<SyntacticLink>, link: SyntacticLink) {
    if !links.iter().any(|existing| {
        existing.left == link.left && existing.right == link.right && existing.kind == link.kind
    }) {
        links.push(link);
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

fn base_pos(word: &str) -> PartOfSpeech {
    let word = normalize_syntax_word(word);
    if is_auxiliary(&word) {
        PartOfSpeech::Auxiliary
    } else if is_determiner(&word) {
        PartOfSpeech::Determiner
    } else if is_preposition(&word) {
        PartOfSpeech::Preposition
    } else if is_coordination_conjunction(&word) {
        PartOfSpeech::Conjunction
    } else if is_subordinating_conjunction(&word) {
        PartOfSpeech::Conjunction
    } else if is_adverb(&word) {
        PartOfSpeech::Adverb
    } else if is_adjective(&word) {
        PartOfSpeech::Adjective
    } else if is_vocative_opener(&word) {
        PartOfSpeech::Particle
    } else if is_proper_name(&word) {
        PartOfSpeech::ProperName
    } else if matches!(
        word.as_str(),
        "i" | "me"
            | "you"
            | "he"
            | "him"
            | "she"
            | "her"
            | "it"
            | "we"
            | "us"
            | "they"
            | "them"
            | "who"
            | "whom"
            | "what"
            | "which"
    ) {
        PartOfSpeech::Pronoun
    } else if is_likely_verb(&word) {
        PartOfSpeech::Verb
    } else {
        PartOfSpeech::Noun
    }
}

fn is_function_word(word: &str) -> bool {
    is_auxiliary(word)
        || is_determiner(word)
        || is_preposition(word)
        || is_coordination_conjunction(word)
        || is_subordinating_conjunction(word)
        || is_complementizer(word)
}

fn is_auxiliary(word: &str) -> bool {
    matches!(
        word,
        "am" | "are"
            | "aren't"
            | "is"
            | "isn't"
            | "was"
            | "wasn't"
            | "were"
            | "weren't"
            | "do"
            | "don't"
            | "does"
            | "doesn't"
            | "did"
            | "didn't"
            | "have"
            | "haven't"
            | "has"
            | "hasn't"
            | "had"
            | "hadn't"
            | "can"
            | "can't"
            | "could"
            | "couldn't"
            | "will"
            | "won't"
            | "would"
            | "wouldn't"
            | "shall"
            | "should"
            | "shouldn't"
            | "may"
            | "might"
            | "must"
            | "ought"
            | "need"
            | "dare"
            | "be"
            | "been"
            | "being"
    )
}

fn is_copula(word: &str) -> bool {
    matches!(
        word,
        "am" | "are"
            | "aren't"
            | "is"
            | "isn't"
            | "was"
            | "wasn't"
            | "were"
            | "weren't"
            | "be"
            | "been"
            | "being"
    )
}

fn is_determiner(word: &str) -> bool {
    matches!(
        word,
        "a" | "an"
            | "the"
            | "this"
            | "that"
            | "these"
            | "those"
            | "my"
            | "your"
            | "our"
            | "their"
            | "his"
            | "her"
            | "its"
            | "all"
            | "another"
            | "any"
            | "both"
            | "each"
            | "either"
            | "every"
            | "many"
            | "much"
            | "no"
            | "some"
            | "such"
            | "what"
            | "which"
    )
}

fn is_preposition(word: &str) -> bool {
    matches!(
        word,
        "about"
            | "above"
            | "across"
            | "after"
            | "against"
            | "along"
            | "around"
            | "at"
            | "before"
            | "behind"
            | "below"
            | "beside"
            | "besides"
            | "between"
            | "by"
            | "during"
            | "for"
            | "from"
            | "in"
            | "inside"
            | "into"
            | "like"
            | "near"
            | "of"
            | "off"
            | "on"
            | "onto"
            | "out"
            | "over"
            | "through"
            | "throughout"
            | "to"
            | "under"
            | "until"
            | "up"
            | "with"
            | "without"
    )
}

fn is_coordination_conjunction(word: &str) -> bool {
    matches!(word, "and" | "or" | "but" | "nor")
}

fn is_subordinating_conjunction(word: &str) -> bool {
    matches!(
        word,
        "after"
            | "although"
            | "as"
            | "because"
            | "before"
            | "if"
            | "since"
            | "though"
            | "unless"
            | "until"
            | "when"
            | "where"
            | "whether"
            | "while"
    )
}

fn is_complementizer(word: &str) -> bool {
    matches!(
        word,
        "that" | "whether" | "if" | "who" | "what" | "which" | "how"
    )
}

fn is_likely_nominal(word: &str) -> bool {
    !is_function_word(word)
        || matches!(
            word,
            "i" | "me"
                | "you"
                | "he"
                | "him"
                | "she"
                | "her"
                | "it"
                | "we"
                | "us"
                | "they"
                | "them"
                | "who"
                | "what"
                | "which"
                | "this"
                | "that"
                | "these"
                | "those"
        )
}

fn is_subject_candidate(word: &str) -> bool {
    is_likely_nominal(word)
        && !is_preposition(word)
        && (!is_modifier_only(word) || is_demonstrative_pronoun(word))
}

fn is_likely_verb(word: &str) -> bool {
    matches!(
        word,
        "act"
            | "appear"
            | "arrive"
            | "ask"
            | "asked"
            | "be"
            | "believe"
            | "bought"
            | "buy"
            | "came"
            | "chase"
            | "chased"
            | "choose"
            | "close"
            | "coming"
            | "come"
            | "comply"
            | "conduct"
            | "console"
            | "contrast"
            | "contrasted"
            | "decide"
            | "did"
            | "die"
            | "do"
            | "eat"
            | "fix"
            | "gave"
            | "give"
            | "go"
            | "goes"
            | "going"
            | "had"
            | "has"
            | "have"
            | "hear"
            | "help"
            | "hit"
            | "hope"
            | "inhale"
            | "inspect"
            | "invite"
            | "know"
            | "knows"
            | "left"
            | "like"
            | "likes"
            | "made"
            | "make"
            | "meet"
            | "met"
            | "object"
            | "operate"
            | "parse"
            | "permit"
            | "present"
            | "produce"
            | "project"
            | "put"
            | "ran"
            | "read"
            | "realize"
            | "rebel"
            | "record"
            | "remember"
            | "result"
            | "refuse"
            | "rose"
            | "run"
            | "runs"
            | "said"
            | "saw"
            | "say"
            | "see"
            | "seem"
            | "seems"
            | "seen"
            | "smiled"
            | "subject"
            | "talk"
            | "tell"
            | "think"
            | "thinks"
            | "thought"
            | "use"
            | "walk"
            | "walked"
            | "want"
            | "wanted"
            | "wants"
            | "went"
            | "win"
            | "wind"
            | "work"
            | "works"
    ) || word.ends_with("ed")
        || word.ends_with("ing")
}

fn is_modifier_pair(left: &str, right: &str) -> bool {
    (is_adjective(left) && is_likely_nominal(right))
        || (is_adverb(left) && (is_adjective(right) || is_likely_verb(right)))
}

fn is_adjective(word: &str) -> bool {
    matches!(
        word,
        "administrative"
            | "afraid"
            | "angry"
            | "beautiful"
            | "big"
            | "black"
            | "bright"
            | "careful"
            | "certain"
            | "clear"
            | "dark"
            | "easy"
            | "excellent"
            | "expensive"
            | "fast"
            | "female"
            | "fortunate"
            | "good"
            | "great"
            | "grotesque"
            | "happy"
            | "heavy"
            | "important"
            | "impatient"
            | "inexpensive"
            | "large"
            | "likely"
            | "long"
            | "lyrical"
            | "medical"
            | "necessary"
            | "new"
            | "obvious"
            | "old"
            | "patient"
            | "possible"
            | "ready"
            | "relaxed"
            | "rude"
            | "short"
            | "slow"
            | "small"
            | "stupid"
            | "sure"
            | "tired"
            | "ugly"
            | "unfortunate"
            | "valid"
            | "white"
    ) || word.ends_with("able")
        || word.ends_with("al")
        || word.ends_with("ful")
        || word.ends_with("ic")
        || word.ends_with("ical")
        || word.ends_with("ive")
        || word.ends_with("less")
        || word.ends_with("ous")
}

fn is_adverb(word: &str) -> bool {
    matches!(
        word,
        "already"
            | "apparently"
            | "broadly"
            | "delicately"
            | "eventually"
            | "fortunately"
            | "generally"
            | "gradually"
            | "initially"
            | "just"
            | "mainly"
            | "never"
            | "not"
            | "now"
            | "often"
            | "particularly"
            | "presumably"
            | "quickly"
            | "really"
            | "recently"
            | "sadly"
            | "sometimes"
            | "soon"
            | "specifically"
            | "straight"
            | "ultimately"
            | "usually"
            | "very"
    ) || word.ends_with("ly")
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

fn is_common_appositive_head(word: &str) -> bool {
    matches!(
        word,
        "actress"
            | "author"
            | "brother"
            | "cousin"
            | "doctor"
            | "expert"
            | "friend"
            | "man"
            | "mother"
            | "president"
            | "singer"
            | "sister"
            | "student"
            | "uncle"
            | "woman"
    )
}

fn is_proper_name(word: &str) -> bool {
    matches!(
        word,
        "abrams"
            | "alfred"
            | "alice"
            | "ann"
            | "anne"
            | "baird"
            | "bob"
            | "chris"
            | "clinton"
            | "david"
            | "dick"
            | "einstein"
            | "emily"
            | "fred"
            | "grace"
            | "janet"
            | "joan"
            | "joe"
            | "john"
            | "ken"
            | "mary"
            | "michael"
            | "nixon"
            | "oj"
            | "rod"
            | "ruth"
            | "sally"
            | "smith"
            | "stuart"
            | "ted"
            | "thomas"
            | "whoopi"
    )
}

fn is_pronoun(word: &str) -> bool {
    matches!(
        word,
        "i" | "me"
            | "you"
            | "he"
            | "him"
            | "she"
            | "her"
            | "it"
            | "we"
            | "us"
            | "they"
            | "them"
            | "who"
            | "whom"
            | "what"
            | "which"
    )
}

fn is_demonstrative_pronoun(word: &str) -> bool {
    matches!(word, "this" | "that" | "these" | "those")
}

fn is_vocative_opener(word: &str) -> bool {
    matches!(word, "hey" | "oh")
}

fn is_parenthetical_marker(word: &str) -> bool {
    matches!(
        word,
        "apparently" | "fortunately" | "however" | "particularly" | "presumably" | "therefore"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};

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

    fn assert_link(analysis: &SentenceSyntaxAnalysis, kind: SyntacticLinkKind) {
        assert!(
            analysis
                .primary_parse()
                .is_some_and(|parse| parse.links.iter().any(|link| link.kind == kind)),
            "expected {kind:?} in {analysis:#?}"
        );
    }

    #[test]
    fn parses_auxiliary_and_coordination_links() {
        let words = ["do", "you", "want", "either", "tea", "or", "coffee"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let analysis = parse_english_link_grammar(&words, Some(TerminalPunctuation::Question));

        assert!(analysis.word_has_link(0, SyntacticLinkKind::Auxiliary));
        assert!(analysis.word_has_link(5, SyntacticLinkKind::Coordination));
        assert!(analysis.matches_environment_pattern(&EnvironmentPattern {
            predicates: vec![ContextPredicate::SyntacticLink(
                SyntacticLinkKind::Coordination
            )],
        }));
    }

    #[test]
    fn upstream_tiny_dict_connector_families_emit_typed_links() {
        // Derived from upstream link-grammar data/en/tiny.dict connector families:
        // D, A/AN, J/Mp/MV, S/O, TO/I, P/AF/C, CO/C.
        let samples = [
            (
                "the small dog chased the cat",
                vec![
                    SyntacticLinkKind::Determiner,
                    SyntacticLinkKind::Modifier,
                    SyntacticLinkKind::Subject,
                    SyntacticLinkKind::Object,
                ],
            ),
            (
                "mary walked out of the room quickly",
                vec![
                    SyntacticLinkKind::Subject,
                    SyntacticLinkKind::Preposition,
                    SyntacticLinkKind::Modifier,
                ],
            ),
            (
                "i want to see the movie",
                vec![
                    SyntacticLinkKind::Subject,
                    SyntacticLinkKind::InfinitivalMarker,
                    SyntacticLinkKind::Object,
                ],
            ),
            (
                "she is very careful about her work",
                vec![
                    SyntacticLinkKind::Subject,
                    SyntacticLinkKind::Modifier,
                    SyntacticLinkKind::Complement,
                    SyntacticLinkKind::Preposition,
                ],
            ),
            (
                "the student and teacher met",
                vec![
                    SyntacticLinkKind::Determiner,
                    SyntacticLinkKind::Coordination,
                    SyntacticLinkKind::Subject,
                ],
            ),
        ];

        for (sentence, expected_links) in samples {
            let analysis = parse_english_link_grammar(&words(sentence), None);
            for expected_link in expected_links {
                assert_link(&analysis, expected_link);
            }
        }
    }

    #[test]
    fn upstream_corpus_basic_samples_cover_nominal_and_clause_rules() {
        // Accepted examples from upstream data/en/corpus-basic.batch. These are
        // fixture-style parity samples, not claims of full Link Grammar parsing.
        let samples = [
            (
                "An income tax increase may be necessary",
                vec![
                    SyntacticLinkKind::Determiner,
                    SyntacticLinkKind::NounCompound,
                    SyntacticLinkKind::Auxiliary,
                    SyntacticLinkKind::Complement,
                ],
            ),
            (
                "This is my friend Bob",
                vec![
                    SyntacticLinkKind::Subject,
                    SyntacticLinkKind::Determiner,
                    SyntacticLinkKind::Complement,
                    SyntacticLinkKind::Apposition,
                ],
            ),
            (
                "I hope that he comes to the party tomorrow",
                vec![
                    SyntacticLinkKind::Subject,
                    SyntacticLinkKind::Complement,
                    SyntacticLinkKind::Preposition,
                    SyntacticLinkKind::Determiner,
                ],
            ),
            (
                "Many people particularly doctors believe there is no health care crisis",
                vec![
                    SyntacticLinkKind::Determiner,
                    SyntacticLinkKind::Parenthetical,
                    SyntacticLinkKind::NounCompound,
                    SyntacticLinkKind::Complement,
                ],
            ),
        ];

        for (sentence, expected_links) in samples {
            let analysis = parse_english_link_grammar(&words(sentence), None);
            for expected_link in expected_links {
                assert_link(&analysis, expected_link);
            }
        }
    }

    #[test]
    fn upstream_ambiguous_verb_lexemes_emit_clause_links() {
        // Classic Link Grammar ambiguous noun/verb examples from data/en/words.
        // The heuristic parser only needs enough of this surface ambiguity to
        // preserve clause structure for downstream prosody rules.
        let samples = [
            (
                "we close the account",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Object],
            ),
            (
                "we conduct the review",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Object],
            ),
            (
                "we console the child",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Object],
            ),
            (
                "we object to the plan",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Preposition],
            ),
            (
                "we permit the request",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Object],
            ),
            (
                "we present the case",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Object],
            ),
            (
                "we produce the record",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Object],
            ),
            (
                "we project the result",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Object],
            ),
            (
                "we rebel against the order",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Preposition],
            ),
            (
                "we refuse the offer",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Object],
            ),
            (
                "we subject the sample to heat",
                vec![
                    SyntacticLinkKind::Subject,
                    SyntacticLinkKind::Object,
                    SyntacticLinkKind::Preposition,
                ],
            ),
            (
                "we wind the clock",
                vec![SyntacticLinkKind::Subject, SyntacticLinkKind::Object],
            ),
        ];

        for (sentence, expected_links) in samples {
            let analysis = parse_english_link_grammar(&words(sentence), None);
            assert_eq!(
                analysis.tokens[1].pos,
                PartOfSpeech::Verb,
                "expected ambiguous lexeme to be usable as verb in {sentence:?}: {analysis:#?}"
            );
            for expected_link in expected_links {
                assert_link(&analysis, expected_link);
            }
        }
    }

    #[test]
    fn emits_vocative_from_upstream_oh_voc_pattern() {
        let analysis = parse_english_link_grammar(&words("Oh Joe listen"), None);

        assert_link(&analysis, SyntacticLinkKind::Vocative);
    }

    #[test]
    #[ignore = "requires the upstream link-parser binary and English dictionary"]
    fn upstream_link_parser_accepts_benchmark_samples() {
        let Ok(dictionary) = std::env::var("LINK_GRAMMAR_EN_DICTIONARY") else {
            eprintln!("set LINK_GRAMMAR_EN_DICTIONARY to upstream data/en to run this comparator");
            return;
        };
        let samples = [
            "The small dog chased the cat",
            "Mary walked out of the room quickly",
            "An income tax increase may be necessary",
            "This is my friend Bob",
            "I hope that he comes to the party tomorrow",
            "Oh Joe listen",
        ];
        let mut child = match Command::new("link-parser")
            .arg("-batch")
            .arg("-verbosity=0")
            .arg(dictionary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                eprintln!("link-parser is not available: {error}");
                return;
            }
        };
        {
            let stdin = child.stdin.as_mut().expect("link-parser stdin");
            for sample in samples {
                writeln!(stdin, "{sample}").expect("write sample to link-parser");
            }
        }

        let output = child
            .wait_with_output()
            .expect("wait for link-parser comparator");
        assert!(
            output.status.success(),
            "link-parser rejected benchmark samples\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
