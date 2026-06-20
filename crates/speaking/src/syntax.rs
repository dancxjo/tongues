use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use crate::data::varieties::DEFAULT_SPEAKING_VARIETY;
use crate::data::varieties::english::syntax as english_syntax;
use crate::data::variety_by_code;
use crate::ids::VarietyId;
use crate::segment::TerminalPunctuation;

pub type WordIndex = usize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SentenceSyntaxAnalysis {
    pub tokens: Vec<SyntaxToken>,
    pub link_parses: Vec<SyntacticLinkParse>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub raw_link_grammar_parses: Vec<RawLinkGrammarParse>,
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
pub struct RawLinkGrammarParse {
    pub links: Vec<RawLinkGrammarLink>,
    pub cost: Option<RawLinkGrammarCost>,
    pub accepted: bool,
    pub backend: RawLinkGrammarBackend,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawLinkGrammarLink {
    pub left: WordIndex,
    pub right: WordIndex,
    pub label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RawLinkGrammarCost {
    pub unused: Option<f32>,
    pub disjunct: Option<f32>,
    pub length: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawLinkGrammarBackend {
    LinkParserCommand,
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
    LinkGrammarProjection,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarietyLinkGrammarParser {
    variety: VarietyId,
}

impl Default for VarietyLinkGrammarParser {
    fn default() -> Self {
        Self::new(VarietyId(DEFAULT_SPEAKING_VARIETY.into()))
    }
}

impl VarietyLinkGrammarParser {
    pub fn new(variety: VarietyId) -> Self {
        Self { variety }
    }

    pub fn variety(&self) -> &VarietyId {
        &self.variety
    }
}

impl LinkGrammarParser for VarietyLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        let Some(variety) = variety_by_code(&self.variety.0) else {
            return SentenceSyntaxAnalysis {
                terminal,
                ..Default::default()
            };
        };
        if let Some(analyzer) = variety.syntax_analyzer {
            return analyzer(words, terminal);
        }
        if let Some(profile) = variety.syntax_heuristics {
            return parse_heuristic_link_grammar(words, terminal, profile);
        }
        SentenceSyntaxAnalysis {
            terminal,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeuristicSyntaxProfile {
    pub determiners: &'static [&'static str],
    pub pronouns: &'static [&'static str],
    pub object_pronouns: &'static [&'static str],
    pub auxiliaries: &'static [&'static str],
    pub copulas: &'static [&'static str],
    pub prepositions: &'static [&'static str],
    pub postpositions: &'static [&'static str],
    pub conjunctions: &'static [&'static str],
    pub particles: &'static [&'static str],
    pub enclitic_suffixes: &'static [&'static str],
    pub complementizers: &'static [&'static str],
    pub adverbs: &'static [&'static str],
    pub adverb_suffixes: &'static [&'static str],
    pub adjectives: &'static [&'static str],
    pub adjective_suffixes: &'static [&'static str],
    pub verbs: &'static [&'static str],
    pub verb_suffixes: &'static [&'static str],
    pub subject_verb_suffixes: &'static [&'static str],
    pub non_verbs: &'static [&'static str],
}

impl HeuristicSyntaxProfile {
    pub const fn empty() -> Self {
        Self {
            determiners: &[],
            pronouns: &[],
            object_pronouns: &[],
            auxiliaries: &[],
            copulas: &[],
            prepositions: &[],
            postpositions: &[],
            conjunctions: &[],
            particles: &[],
            enclitic_suffixes: &[],
            complementizers: &[],
            adverbs: &[],
            adverb_suffixes: &[],
            adjectives: &[],
            adjective_suffixes: &[],
            verbs: &[],
            verb_suffixes: &[],
            subject_verb_suffixes: &[],
            non_verbs: &[],
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct LinkParserCommandBackend {
    command: Option<String>,
    dictionary: Option<String>,
}

impl LinkParserCommandBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_command(command: impl Into<String>) -> Self {
        Self {
            command: Some(command.into()),
            dictionary: None,
        }
    }

    pub fn with_dictionary(mut self, dictionary: impl Into<String>) -> Self {
        self.dictionary = Some(dictionary.into());
        self
    }

    pub fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> Option<SentenceSyntaxAnalysis> {
        parse_with_link_parser_command(
            words,
            terminal,
            self.command.as_deref(),
            self.dictionary.as_deref(),
        )
    }
}

pub fn parse_builtin_link_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
) -> SentenceSyntaxAnalysis {
    if use_link_parser_command_backend() {
        if let Some(analysis) = LinkParserCommandBackend::new().parse(words, terminal) {
            return analysis;
        }
    }
    parse_builtin_heuristic_link_grammar(words, terminal)
}

pub fn parse_builtin_heuristic_link_grammar(
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

fn parse_multilingual_link_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    profile: HeuristicSyntaxProfile,
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
    let links = build_multilingual_links(&normalized, profile);
    let parse = SyntacticLinkParse { links, rank: 0.72 };
    let tokens = normalized
        .iter()
        .enumerate()
        .map(|(word_index, word)| {
            let previous = word_index
                .checked_sub(1)
                .and_then(|index| normalized.get(index))
                .map(String::as_str);
            let mut syntactic_links = parse
                .links
                .iter()
                .filter_map(|link| {
                    (link.left == word_index || link.right == word_index).then_some(link.kind)
                })
                .collect::<Vec<_>>();
            syntactic_links.sort_unstable_by_key(|kind| *kind as u8);
            syntactic_links.dedup();
            let pos = multilingual_pos(profile, word, previous);
            SyntaxToken {
                word_index,
                text: pairs[word_index].0.clone(),
                pos,
                prosodic_role: multilingual_prosodic_role(pos, &syntactic_links),
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

pub fn parse_heuristic_link_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    profile: HeuristicSyntaxProfile,
) -> SentenceSyntaxAnalysis {
    parse_multilingual_link_grammar(words, terminal, profile)
}

fn build_multilingual_links(
    words: &[String],
    profile: HeuristicSyntaxProfile,
) -> Vec<SyntacticLink> {
    let mut links = Vec::new();
    for (index, window) in words.windows(2).enumerate() {
        let left = window[0].as_str();
        let right = window[1].as_str();
        let previous = index
            .checked_sub(1)
            .and_then(|previous| words.get(previous))
            .map(String::as_str);
        if multilingual_is_determiner(profile, left) && multilingual_is_nominal(profile, right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Determiner, 0.78),
            );
        }
        if multilingual_is_pronoun(profile, left)
            && multilingual_is_likely_verb(profile, right, Some(left))
        {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Subject, 0.77),
            );
        }
        if multilingual_is_auxiliary(profile, left)
            && multilingual_is_likely_verb(profile, right, Some(left))
        {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Auxiliary, 0.76),
            );
        }
        if multilingual_is_preposition(profile, left) && multilingual_is_nominal(profile, right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Preposition, 0.76),
            );
        }
        if multilingual_is_nominal(profile, left) && multilingual_is_postposition(profile, right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Preposition, 0.73),
            );
        }
        if multilingual_is_adverb(profile, left)
            && (multilingual_is_likely_verb(profile, right, Some(left))
                || multilingual_is_adjective(profile, right))
        {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Modifier, 0.66),
            );
        }
        if multilingual_is_conjunction(profile, left) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Coordination, 0.68),
            );
        }
        if multilingual_is_particle(profile, right)
            && (multilingual_is_nominal(profile, left)
                || multilingual_is_likely_verb(profile, left, previous))
        {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Modifier, 0.56),
            );
        }
        if multilingual_is_object_pronoun(profile, left)
            && multilingual_is_likely_verb(profile, right, Some(left))
        {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Object, 0.67),
            );
        }
        if multilingual_is_likely_verb(profile, left, previous)
            && multilingual_is_nominal(profile, right)
            && !multilingual_is_preposition(profile, right)
        {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Object, 0.66),
            );
        }
    }
    push_multilingual_determiner_phrase_links(words, profile, &mut links);
    push_multilingual_modifier_phrase_links(words, profile, &mut links);
    push_multilingual_adposition_links(words, profile, &mut links);
    push_multilingual_auxiliary_links(words, profile, &mut links);
    push_multilingual_complement_links(words, profile, &mut links);
    push_multilingual_relative_clause_links(words, profile, &mut links);
    push_multilingual_coordination_links(words, profile, &mut links);
    push_multilingual_object_pronoun_links(words, profile, &mut links);
    for predicate_index in 0..words.len() {
        let previous = predicate_index
            .checked_sub(1)
            .and_then(|index| words.get(index))
            .map(String::as_str);
        if !multilingual_is_likely_verb(profile, &words[predicate_index], previous)
            && !multilingual_is_auxiliary(profile, &words[predicate_index])
        {
            continue;
        }
        if let Some(subject_index) = multilingual_subject_before(words, profile, predicate_index) {
            push_link(
                &mut links,
                link(
                    subject_index,
                    predicate_index,
                    SyntacticLinkKind::Subject,
                    0.72,
                ),
            );
            if let Some(object_index) = (subject_index + 1..predicate_index)
                .rev()
                .find(|index| multilingual_is_object_candidate(profile, &words[*index]))
            {
                push_link(
                    &mut links,
                    link(
                        object_index,
                        predicate_index,
                        SyntacticLinkKind::Object,
                        0.63,
                    ),
                );
            }
        }
        if !multilingual_is_copula(profile, &words[predicate_index]) {
            if let Some(object_index) = words
                .iter()
                .enumerate()
                .skip(predicate_index + 1)
                .take(5)
                .take_while(|(_, word)| {
                    !multilingual_is_complementizer(profile, word)
                        && !multilingual_is_conjunction(profile, word)
                })
                .find_map(|(index, word)| {
                    multilingual_is_nominal_head(profile, word).then_some(index)
                })
            {
                push_link(
                    &mut links,
                    link(
                        predicate_index,
                        object_index,
                        SyntacticLinkKind::Object,
                        0.66,
                    ),
                );
            }
        }
    }
    links
}

fn multilingual_subject_before(
    words: &[String],
    profile: HeuristicSyntaxProfile,
    predicate_index: usize,
) -> Option<usize> {
    let start = predicate_index.saturating_sub(6);
    (start..predicate_index)
        .rev()
        .find(|index| multilingual_is_subject_pronoun(profile, &words[*index]))
        .or_else(|| {
            (start..predicate_index).find(|index| multilingual_is_subject(profile, &words[*index]))
        })
}

fn push_multilingual_determiner_phrase_links(
    words: &[String],
    profile: HeuristicSyntaxProfile,
    links: &mut Vec<SyntacticLink>,
) {
    for determiner_index in 0..words.len() {
        if !multilingual_is_determiner(profile, &words[determiner_index]) {
            continue;
        }
        if let Some(head_index) = words
            .iter()
            .enumerate()
            .skip(determiner_index + 1)
            .take(4)
            .find_map(|(index, word)| multilingual_is_nominal_head(profile, word).then_some(index))
        {
            push_link(
                links,
                link(
                    determiner_index,
                    head_index,
                    SyntacticLinkKind::Determiner,
                    0.78,
                ),
            );
        }
    }
}

fn push_multilingual_modifier_phrase_links(
    words: &[String],
    profile: HeuristicSyntaxProfile,
    links: &mut Vec<SyntacticLink>,
) {
    for modifier_index in 0..words.len() {
        let word = &words[modifier_index];
        if multilingual_is_adjective(profile, word) {
            if let Some(head_index) = words
                .iter()
                .enumerate()
                .skip(modifier_index + 1)
                .take(3)
                .find_map(|(index, word)| {
                    multilingual_is_nominal_head(profile, word).then_some(index)
                })
                .or_else(|| {
                    (0..modifier_index)
                        .rev()
                        .take(3)
                        .find(|index| multilingual_is_nominal_head(profile, &words[*index]))
                })
            {
                push_link(
                    links,
                    link(
                        modifier_index,
                        head_index,
                        SyntacticLinkKind::Modifier,
                        0.65,
                    ),
                );
            }
        }
        if multilingual_is_adverb(profile, word) {
            if let Some(head_index) = words
                .iter()
                .enumerate()
                .skip(modifier_index + 1)
                .take(4)
                .find_map(|(index, word)| {
                    let previous = index
                        .checked_sub(1)
                        .and_then(|previous| words.get(previous))
                        .map(String::as_str);
                    (multilingual_is_likely_verb(profile, word, previous)
                        || multilingual_is_adjective(profile, word))
                    .then_some(index)
                })
                .or_else(|| {
                    (0..modifier_index).rev().take(4).find(|index| {
                        let previous = index
                            .checked_sub(1)
                            .and_then(|previous| words.get(previous))
                            .map(String::as_str);
                        multilingual_is_likely_verb(profile, &words[*index], previous)
                            || multilingual_is_adjective(profile, &words[*index])
                    })
                })
            {
                push_link(
                    links,
                    link(
                        head_index,
                        modifier_index,
                        SyntacticLinkKind::Modifier,
                        0.64,
                    ),
                );
            }
        }
    }
    for complementizer_index in 0..words.len() {
        if !multilingual_is_complementizer(profile, &words[complementizer_index]) {
            continue;
        }
        if let Some(predicate_index) = words
            .iter()
            .enumerate()
            .skip(complementizer_index + 1)
            .take(5)
            .find_map(|(index, word)| {
                let previous = index
                    .checked_sub(1)
                    .and_then(|previous| words.get(previous))
                    .map(String::as_str);
                multilingual_is_likely_verb(profile, word, previous).then_some(index)
            })
        {
            push_link(
                links,
                link(
                    complementizer_index,
                    predicate_index,
                    SyntacticLinkKind::Complement,
                    0.62,
                ),
            );
        }
    }
}

fn push_multilingual_adposition_links(
    words: &[String],
    profile: HeuristicSyntaxProfile,
    links: &mut Vec<SyntacticLink>,
) {
    for adposition_index in 0..words.len() {
        if multilingual_is_preposition(profile, &words[adposition_index]) {
            if let Some(object_index) = words
                .iter()
                .enumerate()
                .skip(adposition_index + 1)
                .take(4)
                .find_map(|(index, word)| {
                    multilingual_is_nominal_head(profile, word).then_some(index)
                })
            {
                push_link(
                    links,
                    link(
                        adposition_index,
                        object_index,
                        SyntacticLinkKind::Preposition,
                        0.76,
                    ),
                );
            }
        }
        if multilingual_is_postposition(profile, &words[adposition_index]) {
            if let Some(object_index) = (0..adposition_index)
                .rev()
                .take(4)
                .find(|index| multilingual_is_nominal_head(profile, &words[*index]))
            {
                push_link(
                    links,
                    link(
                        object_index,
                        adposition_index,
                        SyntacticLinkKind::Preposition,
                        0.73,
                    ),
                );
            }
        }
    }
}

fn push_multilingual_auxiliary_links(
    words: &[String],
    profile: HeuristicSyntaxProfile,
    links: &mut Vec<SyntacticLink>,
) {
    for auxiliary_index in 0..words.len() {
        if !multilingual_is_auxiliary(profile, &words[auxiliary_index]) {
            continue;
        }
        if let Some(verb_index) = words
            .iter()
            .enumerate()
            .skip(auxiliary_index + 1)
            .take(5)
            .find_map(|(index, word)| {
                multilingual_is_likely_verb(profile, word, Some(&words[auxiliary_index]))
                    .then_some(index)
            })
        {
            push_link(
                links,
                link(
                    auxiliary_index,
                    verb_index,
                    SyntacticLinkKind::Auxiliary,
                    0.74,
                ),
            );
        }
    }
}

fn push_multilingual_complement_links(
    words: &[String],
    profile: HeuristicSyntaxProfile,
    links: &mut Vec<SyntacticLink>,
) {
    for predicate_index in 0..words.len() {
        let previous = predicate_index
            .checked_sub(1)
            .and_then(|index| words.get(index))
            .map(String::as_str);
        if multilingual_is_copula(profile, &words[predicate_index]) {
            if let Some(complement_index) = words
                .iter()
                .enumerate()
                .skip(predicate_index + 1)
                .take(5)
                .find_map(|(index, word)| {
                    (multilingual_is_nominal_head(profile, word)
                        || multilingual_is_adjective(profile, word))
                    .then_some(index)
                })
            {
                push_link(
                    links,
                    link(
                        predicate_index,
                        complement_index,
                        SyntacticLinkKind::Complement,
                        0.72,
                    ),
                );
            }
        }
        if multilingual_is_likely_verb(profile, &words[predicate_index], previous)
            || multilingual_is_auxiliary(profile, &words[predicate_index])
        {
            if let Some(complementizer_index) = words
                .iter()
                .enumerate()
                .skip(predicate_index + 1)
                .take(6)
                .find_map(|(index, word)| {
                    multilingual_is_complementizer(profile, word).then_some(index)
                })
            {
                push_link(
                    links,
                    link(
                        predicate_index,
                        complementizer_index,
                        SyntacticLinkKind::Complement,
                        0.66,
                    ),
                );
            }
        }
    }
}

fn push_multilingual_relative_clause_links(
    words: &[String],
    profile: HeuristicSyntaxProfile,
    links: &mut Vec<SyntacticLink>,
) {
    for marker_index in 1..words.len() {
        let marker = &words[marker_index];
        if !multilingual_is_relative_marker(profile, marker) {
            continue;
        }
        let head_index = marker_index - 1;
        if !multilingual_is_nominal_head(profile, &words[head_index]) {
            continue;
        }
        if let Some(predicate_index) = words
            .iter()
            .enumerate()
            .skip(marker_index + 1)
            .take(6)
            .find_map(|(index, word)| {
                let previous = index
                    .checked_sub(1)
                    .and_then(|previous| words.get(previous))
                    .map(String::as_str);
                multilingual_is_likely_verb(profile, word, previous).then_some(index)
            })
        {
            push_link(
                links,
                link(
                    head_index,
                    marker_index,
                    SyntacticLinkKind::Apposition,
                    0.58,
                ),
            );
            push_link(
                links,
                link(
                    marker_index,
                    predicate_index,
                    SyntacticLinkKind::Complement,
                    0.61,
                ),
            );
        }
    }
}

fn push_multilingual_coordination_links(
    words: &[String],
    profile: HeuristicSyntaxProfile,
    links: &mut Vec<SyntacticLink>,
) {
    for conjunction_index in 1..words.len().saturating_sub(1) {
        if !multilingual_is_conjunction(profile, &words[conjunction_index]) {
            continue;
        }
        push_link(
            links,
            link(
                conjunction_index - 1,
                conjunction_index + 1,
                SyntacticLinkKind::Coordination,
                0.7,
            ),
        );
        push_link(
            links,
            link(
                conjunction_index,
                conjunction_index + 1,
                SyntacticLinkKind::Coordination,
                0.68,
            ),
        );
    }
    for particle_index in 1..words.len() {
        if multilingual_is_particle(profile, &words[particle_index])
            && multilingual_is_conjunction(profile, &words[particle_index])
        {
            push_link(
                links,
                link(
                    particle_index - 1,
                    particle_index,
                    SyntacticLinkKind::Coordination,
                    0.62,
                ),
            );
        }
    }
}

fn push_multilingual_object_pronoun_links(
    words: &[String],
    profile: HeuristicSyntaxProfile,
    links: &mut Vec<SyntacticLink>,
) {
    for pronoun_index in 0..words.len() {
        if !multilingual_is_object_pronoun(profile, &words[pronoun_index]) {
            continue;
        }
        if let Some(verb_index) = words
            .iter()
            .enumerate()
            .skip(pronoun_index + 1)
            .take(4)
            .find_map(|(index, word)| {
                let previous = index
                    .checked_sub(1)
                    .and_then(|previous| words.get(previous))
                    .map(String::as_str);
                multilingual_is_likely_verb(profile, word, previous).then_some(index)
            })
            .or_else(|| {
                (0..pronoun_index).rev().take(3).find(|index| {
                    let previous = index
                        .checked_sub(1)
                        .and_then(|previous| words.get(previous))
                        .map(String::as_str);
                    multilingual_is_likely_verb(profile, &words[*index], previous)
                })
            })
        {
            push_link(
                links,
                link(pronoun_index, verb_index, SyntacticLinkKind::Object, 0.67),
            );
        }
    }
}

fn multilingual_pos(
    profile: HeuristicSyntaxProfile,
    word: &str,
    previous: Option<&str>,
) -> PartOfSpeech {
    if multilingual_is_auxiliary(profile, word) {
        PartOfSpeech::Auxiliary
    } else if multilingual_is_determiner(profile, word) {
        PartOfSpeech::Determiner
    } else if multilingual_is_preposition(profile, word) {
        PartOfSpeech::Preposition
    } else if multilingual_is_pronoun(profile, word) {
        PartOfSpeech::Pronoun
    } else if multilingual_is_conjunction(profile, word) {
        PartOfSpeech::Conjunction
    } else if multilingual_is_adverb(profile, word) {
        PartOfSpeech::Adverb
    } else if multilingual_is_likely_verb(profile, word, previous) {
        PartOfSpeech::Verb
    } else if multilingual_is_adjective(profile, word) {
        PartOfSpeech::Adjective
    } else if multilingual_is_nominal(profile, word) {
        PartOfSpeech::Noun
    } else {
        PartOfSpeech::Unknown
    }
}

fn multilingual_prosodic_role(pos: PartOfSpeech, links: &[SyntacticLinkKind]) -> ProsodicRole {
    if links.contains(&SyntacticLinkKind::Object) || links.contains(&SyntacticLinkKind::Complement)
    {
        ProsodicRole::Focus
    } else if matches!(
        pos,
        PartOfSpeech::Auxiliary
            | PartOfSpeech::Determiner
            | PartOfSpeech::Preposition
            | PartOfSpeech::Pronoun
            | PartOfSpeech::Conjunction
            | PartOfSpeech::Particle
    ) {
        ProsodicRole::FunctionWeak
    } else {
        ProsodicRole::Content
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

fn use_link_parser_command_backend() -> bool {
    match std::env::var("TONGUES_LINK_GRAMMAR_BACKEND") {
        Ok(value)
            if matches!(
                value.to_ascii_lowercase().as_str(),
                "heuristic" | "off" | "false" | "0"
            ) =>
        {
            false
        }
        Ok(value)
            if matches!(
                value.to_ascii_lowercase().as_str(),
                "command" | "link-parser"
            ) =>
        {
            true
        }
        _ => link_parser_command_available(),
    }
}

fn link_parser_command_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let command = std::env::var("LINK_GRAMMAR_PARSER")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "link-parser".to_string());
        Command::new(command)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    })
}

fn parse_with_link_parser_command(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    command: Option<&str>,
    dictionary: Option<&str>,
) -> Option<SentenceSyntaxAnalysis> {
    let normalized_words = words
        .iter()
        .filter_map(|word| {
            let normalized = normalize_syntax_word(word);
            (!normalized.is_empty()).then(|| (word.clone(), normalized))
        })
        .collect::<Vec<_>>();
    if normalized_words.is_empty() {
        return Some(SentenceSyntaxAnalysis {
            terminal,
            ..Default::default()
        });
    }

    let sentence = link_parser_sentence(words, terminal);
    let command = command
        .map(str::to_string)
        .or_else(|| std::env::var("LINK_GRAMMAR_PARSER").ok())
        .unwrap_or_else(|| "link-parser".to_string());
    let dictionary = dictionary
        .map(str::to_string)
        .or_else(|| std::env::var("LINK_GRAMMAR_EN_DICTIONARY").ok());

    let mut process = Command::new(command);
    if let Some(dictionary) = dictionary.filter(|value| !value.trim().is_empty()) {
        process.arg(dictionary);
    }
    process
        .arg("--quiet")
        .arg("-verbosity=0")
        .arg("-graphics=1")
        .arg("-links=1")
        .arg("-limit=1")
        .arg("-timeout=5")
        .arg("-width=16381")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = process.spawn().ok()?;
    {
        let stdin = child.stdin.as_mut()?;
        writeln!(stdin, "{sentence}").ok()?;
        writeln!(stdin, "!exit").ok()?;
    }
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    analysis_from_link_parser_output(&stdout, &normalized_words, terminal)
}

fn link_parser_sentence(words: &[String], terminal: Option<TerminalPunctuation>) -> String {
    let mut sentence = words.join(" ");
    let punctuation = match terminal {
        Some(TerminalPunctuation::Question) => Some('?'),
        Some(TerminalPunctuation::Exclamation) => Some('!'),
        Some(TerminalPunctuation::Period) => Some('.'),
        None => None,
    };
    if let Some(punctuation) = punctuation {
        if !sentence.ends_with(['.', '?', '!']) {
            sentence.push(punctuation);
        }
    }
    sentence
}

fn analysis_from_link_parser_output(
    output: &str,
    original_normalized_words: &[(String, String)],
    terminal: Option<TerminalPunctuation>,
) -> Option<SentenceSyntaxAnalysis> {
    let lines = output.lines().collect::<Vec<_>>();
    let word_line_index = link_parser_word_line_index(&lines, original_normalized_words)?;
    let word_positions =
        link_parser_word_positions(lines[word_line_index], original_normalized_words)?;
    let mut raw_links = Vec::new();
    for line in lines[..word_line_index].iter().rev() {
        if line.trim_start().starts_with("Linkage ") {
            break;
        }
        raw_links.extend(parse_link_parser_arc_line(line, &word_positions));
    }
    raw_links.sort_by(|left, right| {
        left.left
            .cmp(&right.left)
            .then(left.right.cmp(&right.right))
            .then(left.label.cmp(&right.label))
    });
    raw_links.dedup_by(|left, right| {
        left.left == right.left && left.right == right.right && left.label == right.label
    });
    if raw_links.is_empty() {
        return None;
    }

    let cost = lines
        .iter()
        .find_map(|line| parse_link_parser_cost_vector(line));
    let raw_parse = RawLinkGrammarParse {
        links: raw_links,
        cost,
        accepted: true,
        backend: RawLinkGrammarBackend::LinkParserCommand,
    };
    Some(project_raw_link_grammar_parse(
        original_normalized_words,
        terminal,
        raw_parse,
    ))
}

fn link_parser_word_line_index(
    lines: &[&str],
    original_normalized_words: &[(String, String)],
) -> Option<usize> {
    lines.iter().position(|line| {
        let normalized_line_words = line
            .split_whitespace()
            .filter_map(normalize_link_parser_output_word)
            .collect::<Vec<_>>();
        original_normalized_words.iter().all(|(_, word)| {
            normalized_line_words
                .iter()
                .any(|candidate| candidate == word)
        })
    })
}

fn normalize_link_parser_output_word(token: &str) -> Option<String> {
    let token = token
        .trim_matches(|character: char| matches!(character, '[' | ']'))
        .split_once('[')
        .map_or(token, |(word, _)| word)
        .split_once('.')
        .map_or(token, |(word, _)| word);
    let normalized = normalize_syntax_word(token);
    (!normalized.is_empty() && normalized != "leftwall" && normalized != "rightwall")
        .then_some(normalized)
}

fn link_parser_word_positions(
    line: &str,
    original_normalized_words: &[(String, String)],
) -> Option<Vec<(usize, usize)>> {
    let mut positions = Vec::new();
    let mut search_start = 0;
    for (word_index, (_, expected)) in original_normalized_words.iter().enumerate() {
        let Some((start, end)) = line[search_start..]
            .split_whitespace()
            .scan(search_start, |offset, token| {
                let start = line[*offset..]
                    .find(token)
                    .map(|relative| *offset + relative)?;
                let end = start + token.len();
                *offset = end;
                Some((token, start, end))
            })
            .find_map(|(token, start, end)| {
                (normalize_link_parser_output_word(token).as_deref() == Some(expected.as_str()))
                    .then_some((start, end))
            })
        else {
            return None;
        };
        positions.push((word_index, (start + end) / 2));
        search_start = end;
    }
    Some(positions)
}

fn parse_link_parser_arc_line(
    line: &str,
    word_positions: &[(usize, usize)],
) -> Vec<RawLinkGrammarLink> {
    let pluses = line
        .char_indices()
        .filter_map(|(index, character)| (character == '+').then_some(index))
        .collect::<Vec<_>>();
    let mut links = Vec::new();
    for window in pluses.windows(2) {
        let left_column = window[0];
        let right_column = window[1];
        let label = line[left_column + 1..right_column]
            .chars()
            .filter(|character| {
                !matches!(character, '-' | '=' | '<' | '>' | '+' | '|' | ' ' | '\t')
            })
            .collect::<String>();
        if label.is_empty() {
            continue;
        }
        let Some(left) = nearest_link_parser_word_index(left_column, word_positions) else {
            continue;
        };
        let Some(right) = nearest_link_parser_word_index(right_column, word_positions) else {
            continue;
        };
        if left == right {
            continue;
        }
        links.push(RawLinkGrammarLink {
            left: left.min(right),
            right: left.max(right),
            label,
        });
    }
    links
}

fn nearest_link_parser_word_index(
    column: usize,
    word_positions: &[(usize, usize)],
) -> Option<usize> {
    word_positions
        .iter()
        .min_by_key(|(_, word_column)| word_column.abs_diff(column))
        .map(|(word_index, _)| *word_index)
}

fn parse_link_parser_cost_vector(line: &str) -> Option<RawLinkGrammarCost> {
    let (_, rest) = line.split_once("cost vector =")?;
    let vector = rest
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect::<Vec<_>>();
    Some(RawLinkGrammarCost {
        unused: parse_cost_component(&vector, "UNUSED"),
        disjunct: parse_cost_component(&vector, "DIS"),
        length: parse_cost_component(&vector, "LEN"),
    })
}

fn parse_cost_component(parts: &[&str], key: &str) -> Option<f32> {
    parts.iter().enumerate().find_map(|(index, part)| {
        if let Some((left, right)) = part.split_once('=') {
            if left == key {
                return right
                    .parse()
                    .ok()
                    .or_else(|| parts.get(index + 1).and_then(|value| value.parse().ok()));
            }
        }
        (part.trim_end_matches('=') == key)
            .then(|| parts.get(index + 1).and_then(|value| value.parse().ok()))
            .flatten()
    })
}

fn project_raw_link_grammar_parse(
    original_normalized_words: &[(String, String)],
    terminal: Option<TerminalPunctuation>,
    raw_parse: RawLinkGrammarParse,
) -> SentenceSyntaxAnalysis {
    let links = raw_parse
        .links
        .iter()
        .filter_map(project_raw_link_grammar_link)
        .collect::<Vec<_>>();
    let rank = raw_parse
        .cost
        .as_ref()
        .map(|cost| {
            1.0 / (1.0
                + cost.unused.unwrap_or_default()
                + cost.disjunct.unwrap_or_default()
                + cost.length.unwrap_or_default())
        })
        .unwrap_or(1.0);
    let parse = SyntacticLinkParse { links, rank };
    let tokens = original_normalized_words
        .iter()
        .enumerate()
        .map(|(word_index, (surface, word))| {
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
                text: surface.clone(),
                pos: disambiguate_pos_from_links(word_index, base_pos(word), &parse.links),
                prosodic_role: prosodic_role_for_word(word, &syntactic_links),
                syntactic_links,
            }
        })
        .collect();

    SentenceSyntaxAnalysis {
        tokens,
        link_parses: vec![parse],
        raw_link_grammar_parses: vec![raw_parse],
        terminal,
    }
}

fn project_raw_link_grammar_link(raw: &RawLinkGrammarLink) -> Option<SyntacticLink> {
    let kind = project_link_grammar_label(&raw.label)?;
    Some(SyntacticLink {
        left: raw.left,
        right: raw.right,
        kind,
        confidence: 0.95,
        source: SyntacticLinkSource::LinkGrammarProjection,
    })
}

fn project_link_grammar_label(label: &str) -> Option<SyntacticLinkKind> {
    let uppercase = label.to_ascii_uppercase();
    if uppercase.starts_with("TO") || uppercase.starts_with('I') {
        Some(SyntacticLinkKind::InfinitivalMarker)
    } else if uppercase.starts_with('S') || uppercase.starts_with("AF") {
        Some(SyntacticLinkKind::Subject)
    } else if uppercase.starts_with('O') {
        Some(SyntacticLinkKind::Object)
    } else if uppercase.starts_with('D') || uppercase.starts_with("YS") {
        Some(SyntacticLinkKind::Determiner)
    } else if uppercase.starts_with("CO")
        || uppercase.starts_with("CP")
        || uppercase.starts_with("CC")
    {
        Some(SyntacticLinkKind::Coordination)
    } else if uppercase.starts_with('J')
        || uppercase.starts_with('P')
        || uppercase.starts_with("MVp")
    {
        Some(SyntacticLinkKind::Preposition)
    } else if uppercase.starts_with('A')
        || uppercase.starts_with('M')
        || uppercase.starts_with("EA")
        || uppercase.starts_with("PH")
    {
        Some(SyntacticLinkKind::Modifier)
    } else if uppercase.starts_with('C')
        || uppercase.starts_with("TH")
        || uppercase.starts_with('R')
        || uppercase.starts_with('B')
    {
        Some(SyntacticLinkKind::Complement)
    } else if uppercase.starts_with('V') {
        Some(SyntacticLinkKind::Vocative)
    } else {
        None
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

    push_prepositional_phrase_links(&words, &mut links);
    push_possessive_links(words, &mut links);
    push_modifier_phrase_links(&words, &mut links);
    push_auxiliary_phrase_links(&words, &mut links);
    push_core_clause_links(&words, &mut links);
    push_complement_links(&words, &mut links);
    push_fronted_clause_marker_links(words, &mut links);
    push_relative_clause_links(words, &mut links);
    push_particle_links(words, &mut links);
    push_passive_participle_links(words, &mut links);
    push_coordination_links(&words, &mut links);
    push_contrast_links(&words, &mut links);
    links
}

fn normalize_syntax_word(word: &str) -> String {
    let mut normalized = String::new();
    for character in word
        .trim_matches(|character: char| !is_syntax_word_character(character))
        .chars()
    {
        let character = match character {
            '\u{2019}' => '\'',
            other => other,
        };
        if is_syntax_word_character(character) {
            normalized.extend(character.to_lowercase());
        }
    }
    normalized
}

fn is_syntax_word_character(character: char) -> bool {
    character.is_alphabetic()
        || character == '\''
        || matches!(character, '\u{0300}'..='\u{036F}' | '\u{0900}'..='\u{094D}')
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
        if !english_syntax::is_contrast_negator(word) {
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
    english_syntax::base_pos(&normalize_syntax_word(word))
}

fn is_function_word(word: &str) -> bool {
    english_syntax::is_function_word(word)
}

fn is_auxiliary(word: &str) -> bool {
    english_syntax::is_auxiliary(word)
}

fn is_copula(word: &str) -> bool {
    english_syntax::is_copula(word)
}

fn is_determiner(word: &str) -> bool {
    english_syntax::is_determiner(word)
}

fn is_preposition(word: &str) -> bool {
    english_syntax::is_preposition(word)
}

fn is_coordination_conjunction(word: &str) -> bool {
    english_syntax::is_coordination_conjunction(word)
}

fn is_subordinating_conjunction(word: &str) -> bool {
    english_syntax::is_subordinating_conjunction(word)
}

fn is_complementizer(word: &str) -> bool {
    english_syntax::is_complementizer(word)
}

fn is_clause_marker(word: &str) -> bool {
    is_complementizer(word) || is_subordinating_conjunction(word)
}

fn is_relative_marker(word: &str) -> bool {
    matches!(word, "that" | "which" | "who" | "whom" | "whose")
}

fn is_likely_nominal(word: &str) -> bool {
    english_syntax::is_likely_nominal(word)
}

fn is_subject_candidate(word: &str) -> bool {
    is_likely_nominal(word)
        && !is_preposition(word)
        && (!is_modifier_only(word) || is_demonstrative_pronoun(word))
}

fn is_likely_verb(word: &str) -> bool {
    english_syntax::is_likely_verb(word)
}

fn is_modifier_pair(left: &str, right: &str) -> bool {
    (is_adjective(left) && is_likely_nominal(right))
        || (is_adverb(left) && (is_adjective(right) || is_likely_verb(right)))
}

fn is_adjective(word: &str) -> bool {
    english_syntax::is_adjective(word)
}

fn is_adverb(word: &str) -> bool {
    english_syntax::is_adverb(word)
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
    english_syntax::is_common_appositive_head(word)
}

fn is_proper_name(word: &str) -> bool {
    english_syntax::is_proper_name(word)
}

fn is_pronoun(word: &str) -> bool {
    english_syntax::is_pronoun(word)
}

fn is_demonstrative_pronoun(word: &str) -> bool {
    english_syntax::is_demonstrative_pronoun(word)
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

fn is_vocative_opener(word: &str) -> bool {
    english_syntax::is_vocative_opener(word)
}

fn is_parenthetical_marker(word: &str) -> bool {
    english_syntax::is_parenthetical_marker(word)
}

fn multilingual_is_determiner(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.determiners)
}

fn multilingual_is_pronoun(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.pronouns)
}

fn multilingual_is_auxiliary(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.auxiliaries)
}

fn multilingual_is_copula(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.copulas)
}

fn multilingual_is_preposition(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.prepositions)
}

fn multilingual_is_postposition(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.postpositions)
}

fn multilingual_is_conjunction(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.conjunctions) || has_enclitic_suffix(word, profile.enclitic_suffixes)
}

fn multilingual_is_particle(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.particles) || has_enclitic_suffix(word, profile.enclitic_suffixes)
}

fn multilingual_is_complementizer(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.complementizers)
}

fn multilingual_is_relative_marker(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    multilingual_is_complementizer(profile, word) || multilingual_is_pronoun(profile, word)
}

fn multilingual_is_object_pronoun(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.object_pronouns)
}

fn multilingual_is_adverb(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.adverbs) || has_suffix(word, profile.adverb_suffixes)
}

fn multilingual_is_adjective(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.adjectives) || has_suffix(word, profile.adjective_suffixes)
}

fn multilingual_is_likely_verb(
    profile: HeuristicSyntaxProfile,
    word: &str,
    previous: Option<&str>,
) -> bool {
    if multilingual_is_auxiliary(profile, word) {
        return true;
    }
    if contains(word, profile.non_verbs) {
        return false;
    }
    contains(word, profile.verbs)
        || has_suffix(word, profile.verb_suffixes)
        || (previous.is_some_and(|previous| multilingual_is_subject_pronoun(profile, previous))
            && has_suffix(word, profile.subject_verb_suffixes))
}

fn multilingual_is_nominal(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    !word.is_empty()
        && !multilingual_is_determiner(profile, word)
        && !multilingual_is_preposition(profile, word)
        && !multilingual_is_postposition(profile, word)
        && !multilingual_is_conjunction(profile, word)
        && !multilingual_is_particle(profile, word)
        && !multilingual_is_complementizer(profile, word)
        && !multilingual_is_adverb(profile, word)
        && !multilingual_is_likely_verb(profile, word, None)
}

fn multilingual_is_nominal_head(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    multilingual_is_nominal(profile, word)
        && !multilingual_is_determiner(profile, word)
        && !multilingual_is_adjective(profile, word)
}

fn multilingual_is_object_candidate(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    multilingual_is_object_pronoun(profile, word) || multilingual_is_nominal_head(profile, word)
}

fn multilingual_is_subject(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    multilingual_is_subject_pronoun(profile, word)
        || (multilingual_is_nominal_head(profile, word)
            && !multilingual_is_object_pronoun(profile, word))
}

fn multilingual_is_subject_pronoun(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    multilingual_is_pronoun(profile, word)
        && !multilingual_is_determiner(profile, word)
        && !multilingual_is_object_pronoun(profile, word)
        && !multilingual_is_complementizer(profile, word)
}

fn has_suffix(word: &str, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|suffix| word.ends_with(suffix))
}

fn has_enclitic_suffix(word: &str, suffixes: &[&str]) -> bool {
    suffixes
        .iter()
        .any(|suffix| word.len() > suffix.len() + 1 && word.ends_with(suffix))
}

fn contains(word: &str, words: &[&str]) -> bool {
    words.contains(&word)
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
                word.trim_matches(|character: char| !is_syntax_word_character(character))
                    .to_string()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    fn parse_variety(code: &str, sentence: &str) -> SentenceSyntaxAnalysis {
        VarietyLinkGrammarParser::new(VarietyId(code.into())).parse(&words(sentence), None)
    }

    fn assert_link(analysis: &SentenceSyntaxAnalysis, kind: SyntacticLinkKind) {
        assert!(
            analysis
                .primary_parse()
                .is_some_and(|parse| parse.links.iter().any(|link| link.kind == kind)),
            "expected {kind:?} in {analysis:#?}"
        );
    }

    fn assert_link_between(
        analysis: &SentenceSyntaxAnalysis,
        left: usize,
        right: usize,
        kind: SyntacticLinkKind,
    ) {
        assert!(
            analysis.primary_parse().is_some_and(|parse| parse
                .links
                .iter()
                .any(|link| link.left == left && link.right == right && link.kind == kind)),
            "expected {kind:?} link {left}->{right} in {analysis:#?}"
        );
    }

    fn assert_no_link_between(
        analysis: &SentenceSyntaxAnalysis,
        left: usize,
        right: usize,
        kind: SyntacticLinkKind,
    ) {
        assert!(
            !analysis.primary_parse().is_some_and(|parse| parse
                .links
                .iter()
                .any(|link| link.left == left && link.right == right && link.kind == kind)),
            "did not expect {kind:?} link {left}->{right} in {analysis:#?}"
        );
    }

    #[test]
    fn parses_link_parser_ascii_output_into_raw_and_projected_links() {
        let original_words = ["the", "dog", "chased", "cat"]
            .into_iter()
            .map(|word| (word.to_string(), word.to_string()))
            .collect::<Vec<_>>();
        let output = r#"
Found 1 linkage (1 had no P.P. violations)
        Linkage 1, cost vector = (UNUSED=0 DIS= 0.00 LEN=8)
    +----Ds---+----Ss----+----Os----+
    |         |          |          |
    the       dog        chased     cat .
"#;

        let analysis = analysis_from_link_parser_output(
            output,
            &original_words,
            Some(TerminalPunctuation::Period),
        )
        .expect("fixture should parse");

        assert_eq!(analysis.raw_link_grammar_parses.len(), 1);
        assert_eq!(
            analysis.raw_link_grammar_parses[0].backend,
            RawLinkGrammarBackend::LinkParserCommand
        );
        assert_eq!(
            analysis.raw_link_grammar_parses[0].cost,
            Some(RawLinkGrammarCost {
                unused: Some(0.0),
                disjunct: Some(0.0),
                length: Some(8.0),
            })
        );
        assert_link_between(&analysis, 0, 1, SyntacticLinkKind::Determiner);
        assert_link_between(&analysis, 1, 2, SyntacticLinkKind::Subject);
        assert_link_between(&analysis, 2, 3, SyntacticLinkKind::Object);
        assert!(analysis.primary_parse().is_some_and(|parse| {
            parse
                .links
                .iter()
                .all(|link| link.source == SyntacticLinkSource::LinkGrammarProjection)
        }));
    }

    #[test]
    fn link_grammar_projection_keeps_unknown_connector_families_raw_only() {
        let original_words = ["left", "right"]
            .into_iter()
            .map(|word| (word.to_string(), word.to_string()))
            .collect::<Vec<_>>();
        let raw_parse = RawLinkGrammarParse {
            links: vec![RawLinkGrammarLink {
                left: 0,
                right: 1,
                label: "ZZcustom".to_string(),
            }],
            cost: None,
            accepted: true,
            backend: RawLinkGrammarBackend::LinkParserCommand,
        };

        let analysis = project_raw_link_grammar_parse(&original_words, None, raw_parse);

        assert_eq!(
            analysis.raw_link_grammar_parses[0].links[0].label,
            "ZZcustom"
        );
        assert!(
            analysis
                .primary_parse()
                .is_some_and(|parse| parse.links.is_empty())
        );
    }

    #[test]
    fn parses_auxiliary_and_coordination_links() {
        let words = ["do", "you", "want", "either", "tea", "or", "coffee"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let analysis = parse_builtin_link_grammar(&words, Some(TerminalPunctuation::Question));

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
            let analysis = parse_builtin_link_grammar(&words(sentence), None);
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
            let analysis = parse_builtin_link_grammar(&words(sentence), None);
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
            let analysis = parse_builtin_link_grammar(&words(sentence), None);
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
    fn english_parser_handles_relative_and_subordinate_clauses() {
        let relative = parse_builtin_link_grammar(&words("the man who saw mary left"), None);
        assert_link_between(&relative, 1, 2, SyntacticLinkKind::Apposition);
        assert_link_between(&relative, 2, 3, SyntacticLinkKind::Complement);
        assert_link_between(&relative, 2, 3, SyntacticLinkKind::Subject);
        assert_link_between(&relative, 3, 4, SyntacticLinkKind::Object);

        let subordinate = parse_builtin_link_grammar(&words("because she left john waited"), None);
        assert_link_between(&subordinate, 0, 2, SyntacticLinkKind::Complement);
        assert_link_between(&subordinate, 1, 2, SyntacticLinkKind::Subject);
        assert_link_between(&subordinate, 3, 4, SyntacticLinkKind::Subject);
    }

    #[test]
    fn english_parser_does_not_promote_clause_subjects_to_matrix_objects() {
        let analysis = parse_builtin_link_grammar(&words("i know that she left"), None);

        assert_link_between(&analysis, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&analysis, 1, 2, SyntacticLinkKind::Complement);
        assert_link_between(&analysis, 2, 4, SyntacticLinkKind::Complement);
        assert_link_between(&analysis, 3, 4, SyntacticLinkKind::Subject);
        assert_no_link_between(&analysis, 1, 3, SyntacticLinkKind::Object);
    }

    #[test]
    fn english_parser_handles_possessives_particles_and_passives() {
        let possessive = parse_builtin_link_grammar(&words("mary's old friend arrived"), None);
        assert_link_between(&possessive, 0, 2, SyntacticLinkKind::Determiner);
        assert_link_between(&possessive, 1, 2, SyntacticLinkKind::Modifier);
        assert_link_between(&possessive, 2, 3, SyntacticLinkKind::Subject);

        let particle = parse_builtin_link_grammar(&words("they turn off the light"), None);
        assert_link_between(&particle, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&particle, 1, 2, SyntacticLinkKind::Modifier);
        assert_link_between(&particle, 1, 4, SyntacticLinkKind::Object);

        let passive = parse_builtin_link_grammar(&words("the ball was thrown by mary"), None);
        assert_link_between(&passive, 1, 3, SyntacticLinkKind::Subject);
        assert_link_between(&passive, 2, 3, SyntacticLinkKind::Auxiliary);
        assert_link_between(&passive, 4, 5, SyntacticLinkKind::Preposition);
    }

    #[test]
    fn english_parser_normalizes_internal_punctuation_and_skips_empty_tokens() {
        let analysis = parse_builtin_link_grammar(
            &["(".into(), "JOHN'S".into(), "old,".into(), "clock!".into()],
            None,
        );

        assert_eq!(analysis.tokens.len(), 3);
        assert_eq!(analysis.tokens[0].text, "JOHN'S");
        assert_link_between(&analysis, 0, 2, SyntacticLinkKind::Determiner);
        assert_link_between(&analysis, 1, 2, SyntacticLinkKind::Modifier);
    }

    #[test]
    fn multilingual_parsers_emit_basic_function_links() {
        let french = parse_variety("fr-FR-Standard", "ils parlent");
        assert_eq!(french.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&french, SyntacticLinkKind::Subject);

        let spanish = parse_variety("es-ES-Castilian", "ellos hablan");
        assert_eq!(spanish.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&spanish, SyntacticLinkKind::Subject);

        let german = parse_variety("de-DE-Standard", "sie sprechen");
        assert_eq!(german.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&german, SyntacticLinkKind::Subject);

        let esperanto = parse_variety("eo", "mi parolas");
        assert_eq!(esperanto.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&esperanto, SyntacticLinkKind::Subject);

        let latin = parse_variety("la-Classical", "ego amo");
        assert_eq!(latin.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&latin, SyntacticLinkKind::Subject);

        let greek = parse_variety("el-GR-Standard", "εγώ λέγω");
        assert_eq!(greek.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&greek, SyntacticLinkKind::Subject);

        let sanskrit = parse_variety("sa-Deva-Standard", "अहम् गच्छति");
        assert_eq!(sanskrit.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&sanskrit, SyntacticLinkKind::Subject);
    }

    #[test]
    fn multilingual_parsers_emit_deeper_phrase_and_clause_links() {
        let french = parse_variety("fr-FR-Standard", "la petite maison sur la colline");
        assert_link_between(&french, 0, 2, SyntacticLinkKind::Determiner);
        assert_link_between(&french, 1, 2, SyntacticLinkKind::Modifier);
        assert_link(&french, SyntacticLinkKind::Preposition);

        let spanish = parse_variety("es-ES-Castilian", "yo veo la casa grande");
        assert_link_between(&spanish, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&spanish, 1, 3, SyntacticLinkKind::Object);
        assert_link_between(&spanish, 4, 3, SyntacticLinkKind::Modifier);

        let german = parse_variety("de-DE-Standard", "ich gebe ihm das buch");
        assert_link_between(&german, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&german, 2, 1, SyntacticLinkKind::Object);
        assert_link_between(&german, 3, 4, SyntacticLinkKind::Determiner);

        let esperanto = parse_variety("eo", "mi estas tre feliĉa");
        assert_link_between(&esperanto, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&esperanto, 1, 3, SyntacticLinkKind::Complement);
        assert_link_between(&esperanto, 3, 2, SyntacticLinkKind::Modifier);
    }

    #[test]
    fn classical_and_indic_parsers_handle_preverbal_objects() {
        let latin = parse_variety("la-Classical", "puella puerum amat");
        assert_link_between(&latin, 0, 2, SyntacticLinkKind::Subject);
        assert_link_between(&latin, 1, 2, SyntacticLinkKind::Object);

        let greek = parse_variety("el-GR-Standard", "εγώ βλέπω τον κόσμο");
        assert_link_between(&greek, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&greek, 1, 3, SyntacticLinkKind::Object);

        let sanskrit = parse_variety("sa-Deva-Standard", "अहं फलम् खादति");
        assert_link_between(&sanskrit, 0, 2, SyntacticLinkKind::Subject);
        assert_link_between(&sanskrit, 1, 2, SyntacticLinkKind::Object);
    }

    #[test]
    fn multilingual_parser_handles_punctuation_clitics_and_empty_tokens() {
        let punctuated = VarietyLinkGrammarParser::new(VarietyId("fr-FR-Standard".into())).parse(
            &["«".into(), "ils,".into(), "parlent!".into(), "»".into()],
            None,
        );
        assert_eq!(punctuated.tokens.len(), 2);
        assert_eq!(punctuated.tokens[0].text, "ils,");
        assert_eq!(punctuated.tokens[1].text, "parlent!");
        assert_link_between(&punctuated, 0, 1, SyntacticLinkKind::Subject);

        let latin = parse_variety("la-Classical", "puella puerque venit");
        assert_link(&latin, SyntacticLinkKind::Coordination);
        assert_link_between(&latin, 0, 2, SyntacticLinkKind::Subject);
    }

    #[test]
    fn multilingual_parser_links_complementizer_clauses() {
        let french = parse_variety("fr-FR-Standard", "je sais que tu viens");
        assert_link_between(&french, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&french, 1, 2, SyntacticLinkKind::Complement);
        assert_link_between(&french, 2, 4, SyntacticLinkKind::Complement);
        assert_no_link_between(&french, 1, 3, SyntacticLinkKind::Object);

        let spanish = parse_variety("es-ES-Castilian", "yo digo que ella viene");
        assert_link_between(&spanish, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&spanish, 1, 2, SyntacticLinkKind::Complement);
        assert_link_between(&spanish, 2, 4, SyntacticLinkKind::Complement);
        assert_no_link_between(&spanish, 1, 3, SyntacticLinkKind::Object);

        let german = parse_variety("de-DE-Standard", "ich weiss dass sie das buch liest");
        assert_link_between(&german, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&german, 1, 2, SyntacticLinkKind::Complement);
        assert_link_between(&german, 2, 6, SyntacticLinkKind::Complement);
        assert_link_between(&german, 3, 6, SyntacticLinkKind::Subject);
        assert_link_between(&german, 5, 6, SyntacticLinkKind::Object);
        assert_no_link_between(&german, 1, 3, SyntacticLinkKind::Object);
    }

    #[test]
    fn multilingual_parser_links_relative_clauses() {
        let french = parse_variety("fr-FR-Standard", "homme qui lit");
        assert_link_between(&french, 0, 1, SyntacticLinkKind::Apposition);
        assert_link_between(&french, 1, 2, SyntacticLinkKind::Complement);

        let spanish = parse_variety("es-ES-Castilian", "hombre que lee");
        assert_link_between(&spanish, 0, 1, SyntacticLinkKind::Apposition);
        assert_link_between(&spanish, 1, 2, SyntacticLinkKind::Complement);

        let german = parse_variety("de-DE-Standard", "mann der liest");
        assert_link_between(&german, 0, 1, SyntacticLinkKind::Apposition);
        assert_link_between(&german, 1, 2, SyntacticLinkKind::Complement);
    }

    #[test]
    fn multilingual_parser_covers_underfit_profile_edges() {
        let french = parse_variety("fr-FR-Standard", "je le vois dans la maison");
        assert_link_between(&french, 0, 2, SyntacticLinkKind::Subject);
        assert_link_between(&french, 1, 2, SyntacticLinkKind::Object);
        assert_link_between(&french, 3, 5, SyntacticLinkKind::Preposition);

        let spanish = parse_variety("es-ES-Castilian", "ella lo ve con el niño");
        assert_link_between(&spanish, 0, 2, SyntacticLinkKind::Subject);
        assert_link_between(&spanish, 1, 2, SyntacticLinkKind::Object);
        assert_link_between(&spanish, 3, 5, SyntacticLinkKind::Preposition);

        let german = parse_variety("de-DE-Standard", "ich bin sehr freundlich");
        assert_link_between(&german, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&german, 1, 3, SyntacticLinkKind::Complement);
        assert_link_between(&german, 3, 2, SyntacticLinkKind::Modifier);

        let esperanto = parse_variety("eo", "mi vidas lin kaj ŝin");
        assert_link_between(&esperanto, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&esperanto, 1, 2, SyntacticLinkKind::Object);
        assert_link(&esperanto, SyntacticLinkKind::Coordination);

        let sanskrit = parse_variety("sa-Deva-Standard", "अहं ग्रामम् मध्ये गच्छति");
        assert_link_between(&sanskrit, 0, 3, SyntacticLinkKind::Subject);
        assert_link_between(&sanskrit, 1, 2, SyntacticLinkKind::Preposition);
    }

    #[test]
    fn french_parser_does_not_treat_common_ent_adjectives_as_verbs() {
        let analysis = parse_variety("fr-FR-Standard", "un homme intelligent");

        assert_ne!(analysis.tokens[2].pos, PartOfSpeech::Verb);
        assert_link(&analysis, SyntacticLinkKind::Determiner);
    }

    #[test]
    fn emits_vocative_from_upstream_oh_voc_pattern() {
        let analysis = parse_builtin_link_grammar(&words("Oh Joe listen"), None);

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
