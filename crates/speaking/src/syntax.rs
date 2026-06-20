use serde::{Deserialize, Serialize};

use crate::data::varieties::{
    english::syntax as english_syntax, esperanto, french, german, greek, latin, sanskrit, spanish,
};
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

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Default, Clone, Copy)]
pub struct EnglishLinkGrammarParser;

impl LinkGrammarParser for EnglishLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        parse_english_link_grammar(words, terminal)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FrenchLinkGrammarParser;

impl LinkGrammarParser for FrenchLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        parse_multilingual_link_grammar(words, terminal, french::syntax_profile())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SpanishLinkGrammarParser;

impl LinkGrammarParser for SpanishLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        parse_multilingual_link_grammar(words, terminal, spanish::syntax_profile())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GermanLinkGrammarParser;

impl LinkGrammarParser for GermanLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        parse_multilingual_link_grammar(words, terminal, german::syntax_profile())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EsperantoLinkGrammarParser;

impl LinkGrammarParser for EsperantoLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        parse_multilingual_link_grammar(words, terminal, esperanto::syntax_profile())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LatinLinkGrammarParser;

impl LinkGrammarParser for LatinLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        parse_multilingual_link_grammar(words, terminal, latin::syntax_profile())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GreekLinkGrammarParser;

impl LinkGrammarParser for GreekLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        parse_multilingual_link_grammar(words, terminal, greek::syntax_profile())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SanskritLinkGrammarParser;

impl LinkGrammarParser for SanskritLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        parse_multilingual_link_grammar(words, terminal, sanskrit::syntax_profile())
    }
}

#[deprecated(note = "use EnglishLinkGrammarParser for English-specific syntax")]
pub type HeuristicLinkGrammarParser = EnglishLinkGrammarParser;

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

fn parse_multilingual_link_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    profile: HeuristicSyntaxProfile,
) -> SentenceSyntaxAnalysis {
    let normalized = words
        .iter()
        .map(|word| normalize_syntax_word(word))
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
                text: words[word_index].clone(),
                pos,
                prosodic_role: multilingual_prosodic_role(pos, &syntactic_links),
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
        if let Some(subject_index) = multilingual_subject_before(words, profile, predicate_index)
        {
            push_link(
                &mut links,
                link(
                    subject_index,
                    predicate_index,
                    SyntacticLinkKind::Subject,
                    0.72,
                ),
            );
            if let Some(object_index) = (subject_index + 1..predicate_index).rev().find(|index| {
                multilingual_is_object_candidate(profile, &words[*index])
            }) {
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
                .find_map(|(index, word)| {
                    multilingual_is_object_candidate(profile, word).then_some(index)
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
        .find(|index| multilingual_is_pronoun(profile, &words[*index]))
        .or_else(|| {
            (start..predicate_index)
                .find(|index| multilingual_is_subject(profile, &words[*index]))
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
    word.trim_matches(|character: char| !is_syntax_word_character(character))
        .chars()
        .flat_map(char::to_lowercase)
        .collect()
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
    contains(word, profile.conjunctions)
}

fn multilingual_is_particle(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.particles)
}

fn multilingual_is_complementizer(profile: HeuristicSyntaxProfile, word: &str) -> bool {
    contains(word, profile.complementizers)
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
        || (previous.is_some_and(|previous| {
            multilingual_is_pronoun(profile, previous)
                && !multilingual_is_determiner(profile, previous)
                && !multilingual_is_object_pronoun(profile, previous)
        })
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
    multilingual_is_pronoun(profile, word)
        || (multilingual_is_nominal_head(profile, word)
            && !multilingual_is_object_pronoun(profile, word))
}

fn has_suffix(word: &str, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|suffix| word.ends_with(suffix))
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
            analysis.primary_parse().is_some_and(|parse| parse.links.iter().any(
                |link| link.left == left && link.right == right && link.kind == kind
            )),
            "expected {kind:?} link {left}->{right} in {analysis:#?}"
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
    fn multilingual_parsers_emit_basic_function_links() {
        let french = FrenchLinkGrammarParser.parse(&words("ils parlent"), None);
        assert_eq!(french.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&french, SyntacticLinkKind::Subject);

        let spanish = SpanishLinkGrammarParser.parse(&words("ellos hablan"), None);
        assert_eq!(spanish.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&spanish, SyntacticLinkKind::Subject);

        let german = GermanLinkGrammarParser.parse(&words("sie sprechen"), None);
        assert_eq!(german.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&german, SyntacticLinkKind::Subject);

        let esperanto = EsperantoLinkGrammarParser.parse(&words("mi parolas"), None);
        assert_eq!(esperanto.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&esperanto, SyntacticLinkKind::Subject);

        let latin = LatinLinkGrammarParser.parse(&words("ego amo"), None);
        assert_eq!(latin.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&latin, SyntacticLinkKind::Subject);

        let greek = GreekLinkGrammarParser.parse(&words("εγώ λέγω"), None);
        assert_eq!(greek.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&greek, SyntacticLinkKind::Subject);

        let sanskrit = SanskritLinkGrammarParser.parse(&words("अहम् गच्छति"), None);
        assert_eq!(sanskrit.tokens[1].pos, PartOfSpeech::Verb);
        assert_link(&sanskrit, SyntacticLinkKind::Subject);
    }

    #[test]
    fn multilingual_parsers_emit_deeper_phrase_and_clause_links() {
        let french = FrenchLinkGrammarParser.parse(&words("la petite maison sur la colline"), None);
        assert_link_between(&french, 0, 2, SyntacticLinkKind::Determiner);
        assert_link_between(&french, 1, 2, SyntacticLinkKind::Modifier);
        assert_link(&french, SyntacticLinkKind::Preposition);

        let spanish = SpanishLinkGrammarParser.parse(&words("yo veo la casa grande"), None);
        assert_link_between(&spanish, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&spanish, 1, 3, SyntacticLinkKind::Object);
        assert_link_between(&spanish, 4, 3, SyntacticLinkKind::Modifier);

        let german = GermanLinkGrammarParser.parse(&words("ich gebe ihm das buch"), None);
        assert_link_between(&german, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&german, 2, 1, SyntacticLinkKind::Object);
        assert_link_between(&german, 3, 4, SyntacticLinkKind::Determiner);

        let esperanto = EsperantoLinkGrammarParser.parse(&words("mi estas tre feliĉa"), None);
        assert_link_between(&esperanto, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&esperanto, 1, 3, SyntacticLinkKind::Complement);
        assert_link_between(&esperanto, 3, 2, SyntacticLinkKind::Modifier);
    }

    #[test]
    fn classical_and_indic_parsers_handle_preverbal_objects() {
        let latin = LatinLinkGrammarParser.parse(&words("puella puerum amat"), None);
        assert_link_between(&latin, 0, 2, SyntacticLinkKind::Subject);
        assert_link_between(&latin, 1, 2, SyntacticLinkKind::Object);

        let greek = GreekLinkGrammarParser.parse(&words("εγώ βλέπω τον κόσμο"), None);
        assert_link_between(&greek, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&greek, 1, 3, SyntacticLinkKind::Object);

        let sanskrit = SanskritLinkGrammarParser.parse(&words("अहं फलम् खादति"), None);
        assert_link_between(&sanskrit, 0, 2, SyntacticLinkKind::Subject);
        assert_link_between(&sanskrit, 1, 2, SyntacticLinkKind::Object);
    }

    #[test]
    fn french_parser_does_not_treat_common_ent_adjectives_as_verbs() {
        let analysis = FrenchLinkGrammarParser.parse(&words("un homme intelligent"), None);

        assert_ne!(analysis.tokens[2].pos, PartOfSpeech::Verb);
        assert_link(&analysis, SyntacticLinkKind::Determiner);
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
