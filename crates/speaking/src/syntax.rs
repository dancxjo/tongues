use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Stdio};

use crate::data::varieties::DEFAULT_SPEAKING_VARIETY;
use crate::data::variety_by_code;
use crate::ids::VarietyId;
use crate::segment::TerminalPunctuation;

pub type WordIndex = usize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SentenceSyntaxAnalysis {
    pub tokens: Vec<SyntaxToken>,
    pub link_parses: Vec<SyntacticLinkParse>,
    /// Raw parser-native dependency/link output. The field name is kept for
    /// wire compatibility with earlier link-grammar-shaped artifacts.
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
    #[serde(alias = "tongues_link_grammar")]
    TonguesRuleGrammar,
    UdPipe,
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
    #[serde(alias = "link_grammar_rule")]
    GrammarRule,
    #[serde(alias = "link_grammar_projection")]
    GrammarProjection,
    UdPipeProjection,
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

pub type GrammarAnalysis = SentenceSyntaxAnalysis;
pub type GrammarToken = SyntaxToken;
pub type GrammarLinkParse = SyntacticLinkParse;
pub type RawGrammarParse = RawLinkGrammarParse;
pub type RawGrammarLink = RawLinkGrammarLink;
pub type RawGrammarCost = RawLinkGrammarCost;
pub type RawGrammarBackend = RawLinkGrammarBackend;
pub type GrammarLink = SyntacticLink;
pub type GrammarLinkKind = SyntacticLinkKind;
pub type GrammarLinkSource = SyntacticLinkSource;
pub type GrammarRuleContext = SyntaxRuleContext;

pub trait GrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis;
}

pub trait LinkGrammarParser: GrammarParser {}

impl<T: GrammarParser + ?Sized> LinkGrammarParser for T {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarParserBackend {
    Auto,
    TonguesRules,
    UdPipe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarietyGrammarParser {
    variety: VarietyId,
    backend: GrammarParserBackend,
}

pub type VarietyLinkGrammarParser = VarietyGrammarParser;

impl Default for VarietyGrammarParser {
    fn default() -> Self {
        Self::new(VarietyId(DEFAULT_SPEAKING_VARIETY.into()))
    }
}

impl VarietyGrammarParser {
    pub fn new(variety: VarietyId) -> Self {
        Self {
            variety,
            backend: GrammarParserBackend::Auto,
        }
    }

    pub fn with_backend(variety: VarietyId, backend: GrammarParserBackend) -> Self {
        Self { variety, backend }
    }

    pub fn variety(&self) -> &VarietyId {
        &self.variety
    }

    pub fn backend(&self) -> GrammarParserBackend {
        self.backend
    }
}

impl GrammarParser for VarietyGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        match self.backend {
            GrammarParserBackend::Auto => {
                if let Some(analysis) = parse_udpipe_for_variety(&self.variety, words, terminal) {
                    return analysis;
                }
                parse_with_variety_rules(&self.variety, words, terminal)
            }
            GrammarParserBackend::TonguesRules => {
                parse_with_variety_rules(&self.variety, words, terminal)
            }
            GrammarParserBackend::UdPipe => {
                parse_udpipe_for_variety(&self.variety, words, terminal).unwrap_or_else(|| {
                    SentenceSyntaxAnalysis {
                        terminal,
                        ..Default::default()
                    }
                })
            }
        }
    }
}

fn parse_with_variety_rules(
    variety_id: &VarietyId,
    words: &[String],
    terminal: Option<TerminalPunctuation>,
) -> SentenceSyntaxAnalysis {
    let Some(variety) = variety_by_code(&variety_id.0) else {
        return SentenceSyntaxAnalysis {
            terminal,
            ..Default::default()
        };
    };
    if let Some(analyzer) = variety.syntax_analyzer {
        return analyzer(words, terminal);
    }
    if let Some(profile) = variety.syntax_rules {
        return parse_grammar_with_rules(words, terminal, profile);
    }
    SentenceSyntaxAnalysis {
        terminal,
        ..Default::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdPipeGrammarParser {
    model_path: String,
    command: String,
}

impl UdPipeGrammarParser {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            command: "udpipe".into(),
        }
    }

    pub fn with_command(model_path: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            command: command.into(),
        }
    }

    pub fn parse_with_status(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> Option<SentenceSyntaxAnalysis> {
        let input = udpipe_horizontal_input(words, terminal);
        let mut child = Command::new(&self.command)
            .arg("--input=horizontal")
            .arg("--tag")
            .arg("--parse")
            .arg(&self.model_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        child.stdin.as_mut()?.write_all(input.as_bytes()).ok()?;
        let output = child.wait_with_output().ok()?;
        if !output.status.success() {
            return None;
        }
        let conllu = String::from_utf8(output.stdout).ok()?;
        analysis_from_udpipe_conllu(words, terminal, &conllu)
    }
}

impl GrammarParser for UdPipeGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        self.parse_with_status(words, terminal)
            .unwrap_or_else(|| SentenceSyntaxAnalysis {
                terminal,
                ..Default::default()
            })
    }
}

fn parse_udpipe_for_variety(
    variety_id: &VarietyId,
    words: &[String],
    terminal: Option<TerminalPunctuation>,
) -> Option<SentenceSyntaxAnalysis> {
    let model_path = udpipe_model_path_for_variety(variety_id)?;
    UdPipeGrammarParser::new(model_path).parse_with_status(words, terminal)
}

fn udpipe_model_path_for_variety(variety_id: &VarietyId) -> Option<String> {
    let scoped = format!(
        "TONGUES_UDPIPE_MODEL_{}",
        variety_id
            .0
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character.to_ascii_uppercase()
                } else {
                    '_'
                }
            })
            .collect::<String>()
    );
    std::env::var(scoped)
        .ok()
        .filter(|path| !path.trim().is_empty())
        .or_else(|| {
            std::env::var("TONGUES_UDPIPE_MODEL")
                .ok()
                .filter(|path| !path.trim().is_empty())
        })
}

fn udpipe_horizontal_input(words: &[String], terminal: Option<TerminalPunctuation>) -> String {
    let mut sentence = words.join(" ");
    match terminal {
        Some(TerminalPunctuation::Question) => sentence.push('?'),
        Some(TerminalPunctuation::Exclamation) => sentence.push('!'),
        Some(TerminalPunctuation::Period) => sentence.push('.'),
        None => {}
    }
    sentence.push('\n');
    sentence
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UdPipeToken {
    id: usize,
    form: String,
    upos: String,
    head: Option<usize>,
    deprel: String,
}

fn analysis_from_udpipe_conllu(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    conllu: &str,
) -> Option<SentenceSyntaxAnalysis> {
    let udpipe_tokens = parse_udpipe_tokens(conllu);
    if udpipe_tokens.is_empty() {
        return None;
    }
    let projected_len = words.len().min(udpipe_tokens.len());
    let mut links = Vec::new();
    let mut raw_links = Vec::new();
    for token in udpipe_tokens.iter().take(projected_len) {
        let Some(head) = token.head else {
            continue;
        };
        if head == 0 || head > projected_len {
            continue;
        }
        let dependent = token.id.saturating_sub(1);
        let head = head - 1;
        if dependent >= projected_len || dependent == head {
            continue;
        }
        let kind = udpipe_deprel_link_kind(&token.deprel);
        let left = dependent.min(head);
        let right = dependent.max(head);
        push_link(
            &mut links,
            SyntacticLink {
                left,
                right,
                kind,
                confidence: 0.9,
                source: SyntacticLinkSource::UdPipeProjection,
            },
        );
        raw_links.push(RawLinkGrammarLink {
            left,
            right,
            label: token.deprel.clone(),
        });
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
    let parse = SyntacticLinkParse { links, rank: 1.0 };
    let tokens = udpipe_tokens
        .iter()
        .take(projected_len)
        .enumerate()
        .map(|(word_index, token)| {
            let syntactic_links = parse
                .links
                .iter()
                .filter(|link| link.left == word_index || link.right == word_index)
                .map(|link| link.kind)
                .collect::<Vec<_>>();
            let pos = udpipe_upos_part_of_speech(&token.upos);
            SyntaxToken {
                word_index,
                text: words
                    .get(word_index)
                    .cloned()
                    .unwrap_or_else(|| token.form.clone()),
                pos,
                prosodic_role: multilingual_prosodic_role(pos, &syntactic_links),
                syntactic_links,
            }
        })
        .collect::<Vec<_>>();
    let unlinked = unlinked_word_count(projected_len, &parse.links) as f32;
    let length = parse
        .links
        .iter()
        .map(|link| link.right.abs_diff(link.left) as f32)
        .sum();
    Some(SentenceSyntaxAnalysis {
        tokens,
        link_parses: vec![parse],
        raw_link_grammar_parses: vec![RawLinkGrammarParse {
            links: raw_links,
            cost: Some(RawLinkGrammarCost {
                unused: Some(unlinked),
                disjunct: None,
                length: Some(length),
            }),
            accepted: true,
            backend: RawLinkGrammarBackend::UdPipe,
        }],
        terminal,
    })
}

fn parse_udpipe_tokens(conllu: &str) -> Vec<UdPipeToken> {
    conllu
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let columns = line.split('\t').collect::<Vec<_>>();
            if columns.len() < 8 || columns[0].contains('-') || columns[0].contains('.') {
                return None;
            }
            Some(UdPipeToken {
                id: columns[0].parse().ok()?,
                form: columns[1].to_string(),
                upos: columns[3].to_string(),
                head: columns[6].parse().ok(),
                deprel: columns[7].to_string(),
            })
        })
        .collect()
}

fn udpipe_upos_part_of_speech(upos: &str) -> PartOfSpeech {
    match upos {
        "NOUN" => PartOfSpeech::Noun,
        "VERB" => PartOfSpeech::Verb,
        "AUX" => PartOfSpeech::Auxiliary,
        "DET" => PartOfSpeech::Determiner,
        "ADP" => PartOfSpeech::Preposition,
        "PRON" => PartOfSpeech::Pronoun,
        "ADV" => PartOfSpeech::Adverb,
        "ADJ" => PartOfSpeech::Adjective,
        "CCONJ" | "SCONJ" => PartOfSpeech::Conjunction,
        "PART" => PartOfSpeech::Particle,
        "PROPN" => PartOfSpeech::ProperName,
        _ => PartOfSpeech::Unknown,
    }
}

fn udpipe_deprel_link_kind(deprel: &str) -> SyntacticLinkKind {
    let base = deprel.split(':').next().unwrap_or(deprel);
    match base {
        "nsubj" | "csubj" => SyntacticLinkKind::Subject,
        "obj" | "iobj" => SyntacticLinkKind::Object,
        "xcomp" | "ccomp" | "acl" | "advcl" => SyntacticLinkKind::Complement,
        "det" => SyntacticLinkKind::Determiner,
        "aux" | "cop" => SyntacticLinkKind::Auxiliary,
        "case" | "obl" => SyntacticLinkKind::Preposition,
        "cc" | "conj" => SyntacticLinkKind::Coordination,
        "compound" | "flat" | "fixed" => SyntacticLinkKind::NounCompound,
        "vocative" => SyntacticLinkKind::Vocative,
        "appos" => SyntacticLinkKind::Apposition,
        "parataxis" | "discourse" => SyntacticLinkKind::Parenthetical,
        _ => SyntacticLinkKind::Modifier,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrammarRuleSet {
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
    pub subject_suffixes: &'static [&'static str],
    pub object_suffixes: &'static [&'static str],
    pub possessive_suffixes: &'static [&'static str],
    pub infinitival_markers: &'static [&'static str],
    pub proper_names: &'static [&'static str],
    pub common_appositive_heads: &'static [&'static str],
    pub vocative_openers: &'static [&'static str],
    pub parenthetical_markers: &'static [&'static str],
    pub contrast_negators: &'static [&'static str],
    pub phrasal_particles: &'static [&'static str],
    pub past_participles: &'static [&'static str],
    pub allow_noun_compounds: bool,
    pub rules: &'static [GrammarRule],
}

impl GrammarRuleSet {
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
            subject_suffixes: &[],
            object_suffixes: &[],
            possessive_suffixes: &[],
            infinitival_markers: &[],
            proper_names: &[],
            common_appositive_heads: &[],
            vocative_openers: &[],
            parenthetical_markers: &[],
            contrast_negators: &[],
            phrasal_particles: &[],
            past_participles: &[],
            allow_noun_compounds: false,
            rules: DEFAULT_GRAMMAR_RULES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrammarRule {
    pub left: GrammarConnector,
    pub right: GrammarConnector,
    pub kind: SyntacticLinkKind,
    pub label: &'static str,
    pub confidence: u8,
    pub max_distance: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarConnector {
    Determiner,
    Nominal,
    NominalHead,
    Subject,
    ObjectPronoun,
    Verb,
    Auxiliary,
    Copula,
    Preposition,
    Postposition,
    Conjunction,
    Particle,
    Complementizer,
    RelativeMarker,
    Adjective,
    Adverb,
}

pub type LinkGrammarRuleSet = GrammarRuleSet;
pub type LinkGrammarRule = GrammarRule;
pub type LinkGrammarConnector = GrammarConnector;

pub const DEFAULT_GRAMMAR_RULES: &[GrammarRule] = &[
    rule(
        GrammarConnector::Determiner,
        GrammarConnector::NominalHead,
        SyntacticLinkKind::Determiner,
        "D",
        78,
        4,
    ),
    rule(
        GrammarConnector::Subject,
        GrammarConnector::Verb,
        SyntacticLinkKind::Subject,
        "S",
        77,
        6,
    ),
    rule(
        GrammarConnector::Auxiliary,
        GrammarConnector::Verb,
        SyntacticLinkKind::Auxiliary,
        "AUX",
        76,
        5,
    ),
    rule(
        GrammarConnector::Preposition,
        GrammarConnector::NominalHead,
        SyntacticLinkKind::Preposition,
        "J",
        76,
        4,
    ),
    rule(
        GrammarConnector::NominalHead,
        GrammarConnector::Postposition,
        SyntacticLinkKind::Preposition,
        "JP",
        73,
        4,
    ),
    rule(
        GrammarConnector::Verb,
        GrammarConnector::NominalHead,
        SyntacticLinkKind::Object,
        "O",
        66,
        5,
    ),
    rule(
        GrammarConnector::ObjectPronoun,
        GrammarConnector::Verb,
        SyntacticLinkKind::Object,
        "O",
        67,
        4,
    ),
    rule(
        GrammarConnector::Adjective,
        GrammarConnector::NominalHead,
        SyntacticLinkKind::Modifier,
        "AN",
        65,
        3,
    ),
    rule(
        GrammarConnector::Verb,
        GrammarConnector::Adverb,
        SyntacticLinkKind::Modifier,
        "MV",
        64,
        4,
    ),
    rule(
        GrammarConnector::Adverb,
        GrammarConnector::Verb,
        SyntacticLinkKind::Modifier,
        "MV",
        66,
        4,
    ),
    rule(
        GrammarConnector::Adverb,
        GrammarConnector::Adjective,
        SyntacticLinkKind::Modifier,
        "EA",
        66,
        4,
    ),
    rule(
        GrammarConnector::Copula,
        GrammarConnector::NominalHead,
        SyntacticLinkKind::Complement,
        "Pa",
        72,
        5,
    ),
    rule(
        GrammarConnector::Copula,
        GrammarConnector::Adjective,
        SyntacticLinkKind::Complement,
        "Pa",
        72,
        5,
    ),
    rule(
        GrammarConnector::Verb,
        GrammarConnector::Complementizer,
        SyntacticLinkKind::Complement,
        "C",
        66,
        6,
    ),
    rule(
        GrammarConnector::Complementizer,
        GrammarConnector::Verb,
        SyntacticLinkKind::Complement,
        "C",
        62,
        5,
    ),
    rule(
        GrammarConnector::NominalHead,
        GrammarConnector::RelativeMarker,
        SyntacticLinkKind::Apposition,
        "R",
        58,
        4,
    ),
    rule(
        GrammarConnector::RelativeMarker,
        GrammarConnector::Verb,
        SyntacticLinkKind::Complement,
        "C",
        61,
        6,
    ),
    rule(
        GrammarConnector::Conjunction,
        GrammarConnector::Nominal,
        SyntacticLinkKind::Coordination,
        "CO",
        68,
        1,
    ),
    rule(
        GrammarConnector::Nominal,
        GrammarConnector::Conjunction,
        SyntacticLinkKind::Coordination,
        "CO",
        62,
        1,
    ),
    rule(
        GrammarConnector::Particle,
        GrammarConnector::Verb,
        SyntacticLinkKind::Modifier,
        "M",
        56,
        1,
    ),
];

const fn rule(
    left: GrammarConnector,
    right: GrammarConnector,
    kind: SyntacticLinkKind,
    label: &'static str,
    confidence: u8,
    max_distance: usize,
) -> GrammarRule {
    GrammarRule {
        left,
        right,
        kind,
        label,
        confidence,
        max_distance,
    }
}

fn parse_rule_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    profile: GrammarRuleSet,
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
    let (links, raw_links) = build_rule_links(&normalized, profile);
    let rank = parse_rank(&links, normalized.len());
    let parse = SyntacticLinkParse { links, rank };
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
            let pos = multilingual_pos_at(profile, &normalized, word_index, &parse.links);
            SyntaxToken {
                word_index,
                text: pairs[word_index].0.clone(),
                pos,
                prosodic_role: multilingual_prosodic_role(pos, &syntactic_links),
                syntactic_links,
            }
        })
        .collect();
    let unlinked = unlinked_word_count(normalized.len(), &parse.links) as f32;
    let length = parse
        .links
        .iter()
        .map(|link| link.right.abs_diff(link.left) as f32)
        .sum();
    let accepted = !parse.links.is_empty() || normalized.is_empty();

    SentenceSyntaxAnalysis {
        tokens,
        link_parses: vec![parse],
        raw_link_grammar_parses: vec![RawLinkGrammarParse {
            links: raw_links,
            cost: Some(RawLinkGrammarCost {
                unused: Some(unlinked),
                disjunct: Some(0.0),
                length: Some(length),
            }),
            accepted,
            backend: RawLinkGrammarBackend::TonguesRuleGrammar,
        }],
        terminal,
    }
}

pub fn parse_grammar_with_rules(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    profile: GrammarRuleSet,
) -> SentenceSyntaxAnalysis {
    parse_rule_grammar(words, terminal, profile)
}

pub const DEFAULT_LINK_GRAMMAR_RULES: &[GrammarRule] = DEFAULT_GRAMMAR_RULES;

pub fn parse_link_grammar_with_rules(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    profile: GrammarRuleSet,
) -> SentenceSyntaxAnalysis {
    parse_grammar_with_rules(words, terminal, profile)
}

fn build_rule_links(
    words: &[String],
    profile: GrammarRuleSet,
) -> (Vec<SyntacticLink>, Vec<RawLinkGrammarLink>) {
    let mut links = Vec::new();
    let mut raw_links = Vec::new();
    apply_connector_rules(words, profile, &mut links, &mut raw_links);
    for (index, window) in words.windows(2).enumerate() {
        let left = window[0].as_str();
        let right = window[1].as_str();
        let previous = index
            .checked_sub(1)
            .and_then(|previous| words.get(previous))
            .map(String::as_str);
        if multilingual_is_determiner(profile, left)
            && !multilingual_is_complementizer_at(profile, words, index)
            && multilingual_is_nominal_at(profile, words, index + 1)
        {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Determiner, 0.78),
            );
        }
        if multilingual_is_subject_pronoun(profile, left)
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
        if multilingual_is_preposition(profile, left)
            && multilingual_is_nominal_at(profile, words, index + 1)
        {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Preposition, 0.76),
            );
        }
        if multilingual_is_nominal_at(profile, words, index)
            && multilingual_is_postposition(profile, right)
        {
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
            && (multilingual_is_nominal_at(profile, words, index)
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
            && multilingual_is_nominal_at(profile, words, index + 1)
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
    push_multilingual_infinitive_links(words, profile, &mut links);
    push_multilingual_possessive_links(words, profile, &mut links);
    push_multilingual_noun_compound_links(words, profile, &mut links);
    push_multilingual_apposition_links(words, profile, &mut links);
    push_multilingual_vocative_links(words, profile, &mut links);
    push_multilingual_parenthetical_links(words, profile, &mut links);
    push_multilingual_particle_links(words, profile, &mut links);
    push_multilingual_passive_links(words, profile, &mut links);
    push_multilingual_contrast_links(words, profile, &mut links);
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
                .take_while(|(index, word)| {
                    !multilingual_is_complementizer_at(profile, words, *index)
                        && !multilingual_is_conjunction(profile, word)
                })
                .find_map(|(index, _)| {
                    multilingual_is_nominal_head_at(profile, words, index).then_some(index)
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
    raw_links.extend(links.iter().map(raw_link_from_typed_link));
    raw_links.sort_by(|left, right| {
        left.left
            .cmp(&right.left)
            .then(left.right.cmp(&right.right))
            .then(left.label.cmp(&right.label))
    });
    raw_links.dedup_by(|left, right| {
        left.left == right.left && left.right == right.right && left.label == right.label
    });
    (links, raw_links)
}

fn apply_connector_rules(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
    raw_links: &mut Vec<RawLinkGrammarLink>,
) {
    for rule in profile.rules {
        for left in 0..words.len() {
            if !connector_matches(profile, rule.left, words, left) {
                continue;
            }
            let right_start = left + 1;
            let right_end = words.len().min(left + rule.max_distance + 1);
            for right in right_start..right_end {
                if !connector_matches(profile, rule.right, words, right) {
                    continue;
                }
                push_link(
                    links,
                    SyntacticLink {
                        left,
                        right,
                        kind: rule.kind,
                        confidence: f32::from(rule.confidence) / 100.0,
                        source: SyntacticLinkSource::GrammarRule,
                    },
                );
                raw_links.push(RawLinkGrammarLink {
                    left,
                    right,
                    label: rule.label.to_string(),
                });
                break;
            }
        }
    }
}

fn connector_matches(
    profile: GrammarRuleSet,
    connector: GrammarConnector,
    words: &[String],
    index: usize,
) -> bool {
    let word = &words[index];
    let previous = index
        .checked_sub(1)
        .and_then(|previous| words.get(previous))
        .map(String::as_str);
    match connector {
        GrammarConnector::Determiner => {
            multilingual_is_determiner(profile, word)
                && !multilingual_is_complementizer_at(profile, words, index)
        }
        GrammarConnector::Nominal => multilingual_is_nominal_at(profile, words, index),
        GrammarConnector::NominalHead => multilingual_is_nominal_head_at(profile, words, index),
        GrammarConnector::Subject => multilingual_is_connector_subject(profile, word),
        GrammarConnector::ObjectPronoun => multilingual_is_object_pronoun(profile, word),
        GrammarConnector::Verb => multilingual_is_likely_verb(profile, word, previous),
        GrammarConnector::Auxiliary => multilingual_is_auxiliary(profile, word),
        GrammarConnector::Copula => multilingual_is_copula(profile, word),
        GrammarConnector::Preposition => multilingual_is_preposition(profile, word),
        GrammarConnector::Postposition => multilingual_is_postposition(profile, word),
        GrammarConnector::Conjunction => multilingual_is_conjunction(profile, word),
        GrammarConnector::Particle => multilingual_is_particle(profile, word),
        GrammarConnector::Complementizer => {
            multilingual_is_complementizer_at(profile, words, index)
        }
        GrammarConnector::RelativeMarker => {
            multilingual_is_relative_marker_at(profile, words, index)
        }
        GrammarConnector::Adjective => multilingual_is_adjective(profile, word),
        GrammarConnector::Adverb => multilingual_is_adverb(profile, word),
    }
}

fn raw_link_from_typed_link(link: &SyntacticLink) -> RawLinkGrammarLink {
    RawLinkGrammarLink {
        left: link.left,
        right: link.right,
        label: grammar_link_label(link.kind).to_string(),
    }
}

fn grammar_link_label(kind: SyntacticLinkKind) -> &'static str {
    match kind {
        SyntacticLinkKind::Subject => "S",
        SyntacticLinkKind::Object => "O",
        SyntacticLinkKind::Complement => "C",
        SyntacticLinkKind::InfinitivalMarker => "TO",
        SyntacticLinkKind::Modifier => "M",
        SyntacticLinkKind::Determiner => "D",
        SyntacticLinkKind::Auxiliary => "AUX",
        SyntacticLinkKind::Preposition => "J",
        SyntacticLinkKind::Coordination => "CO",
        SyntacticLinkKind::ContrastPair => "NEG",
        SyntacticLinkKind::NounCompound => "NN",
        SyntacticLinkKind::Vocative => "VOC",
        SyntacticLinkKind::Apposition => "APP",
        SyntacticLinkKind::Parenthetical => "PAR",
    }
}

fn parse_rank(links: &[SyntacticLink], word_count: usize) -> f32 {
    if word_count == 0 {
        return 1.0;
    }
    let average_confidence =
        links.iter().map(|link| link.confidence).sum::<f32>() / links.len().max(1) as f32;
    let coverage = 1.0 - (unlinked_word_count(word_count, links) as f32 / word_count as f32);
    (average_confidence * 0.7 + coverage * 0.3).clamp(0.0, 1.0)
}

fn unlinked_word_count(word_count: usize, links: &[SyntacticLink]) -> usize {
    let mut linked = HashSet::new();
    for link in links {
        linked.insert(link.left);
        linked.insert(link.right);
    }
    word_count.saturating_sub(linked.len())
}

fn multilingual_subject_before(
    words: &[String],
    profile: GrammarRuleSet,
    predicate_index: usize,
) -> Option<usize> {
    let start = (0..predicate_index)
        .rev()
        .find(|index| {
            let previous = index
                .checked_sub(1)
                .and_then(|previous| words.get(previous))
                .map(String::as_str);
            multilingual_is_likely_verb(profile, &words[*index], previous)
                || multilingual_is_complementizer_at(profile, words, *index)
                || multilingual_is_conjunction(profile, &words[*index])
        })
        .map_or(predicate_index.saturating_sub(6), |index| index + 1);
    (start..predicate_index)
        .rev()
        .find(|index| multilingual_is_subject_pronoun(profile, &words[*index]))
        .or_else(|| {
            (start..predicate_index)
                .rev()
                .find(|index| multilingual_is_subject(profile, &words[*index]))
        })
}

fn push_multilingual_determiner_phrase_links(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    for determiner_index in 0..words.len() {
        if !multilingual_is_determiner(profile, &words[determiner_index])
            || multilingual_is_complementizer_at(profile, words, determiner_index)
        {
            continue;
        }
        if let Some(head_index) = words
            .iter()
            .enumerate()
            .skip(determiner_index + 1)
            .take(4)
            .find_map(|(index, _)| {
                multilingual_is_nominal_head_at(profile, words, index).then_some(index)
            })
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
    profile: GrammarRuleSet,
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
                .find_map(|(index, _)| {
                    multilingual_is_nominal_head_at(profile, words, index).then_some(index)
                })
                .or_else(|| {
                    (0..modifier_index)
                        .rev()
                        .take(3)
                        .find(|index| multilingual_is_nominal_head_at(profile, words, *index))
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
        if !multilingual_is_complementizer_at(profile, words, complementizer_index) {
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
    profile: GrammarRuleSet,
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
                    multilingual_is_nominal_head_at(profile, words, index).then_some(index)
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
                .find(|index| multilingual_is_nominal_head_at(profile, words, *index))
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
    profile: GrammarRuleSet,
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
    profile: GrammarRuleSet,
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
                    (multilingual_is_nominal_head_at(profile, words, index)
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
                .find_map(|(index, _)| {
                    multilingual_is_complementizer_at(profile, words, index).then_some(index)
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
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    for marker_index in 1..words.len() {
        let marker = &words[marker_index];
        if !multilingual_is_relative_marker_at(profile, words, marker_index) {
            continue;
        }
        let head_index = marker_index - 1;
        if !multilingual_is_nominal_head_at(profile, words, head_index) {
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
            if multilingual_is_pronoun(profile, marker) {
                push_link(
                    links,
                    link(
                        marker_index,
                        predicate_index,
                        SyntacticLinkKind::Subject,
                        0.64,
                    ),
                );
            }
        }
    }
}

fn push_multilingual_infinitive_links(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    for marker_index in 0..words.len().saturating_sub(1) {
        if !contains(&words[marker_index], profile.infinitival_markers) {
            continue;
        }
        if let Some(verb_index) = words
            .iter()
            .enumerate()
            .skip(marker_index + 1)
            .take(3)
            .find_map(|(index, word)| {
                multilingual_is_likely_verb(profile, word, Some(&words[marker_index]))
                    .then_some(index)
            })
        {
            push_link(
                links,
                link(
                    marker_index,
                    verb_index,
                    SyntacticLinkKind::InfinitivalMarker,
                    0.9,
                ),
            );
        }
    }
}

fn push_multilingual_possessive_links(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    for possessive_index in 0..words.len().saturating_sub(1) {
        if !is_possessive_nominal(profile, &words[possessive_index]) {
            continue;
        }
        if let Some(head_index) = words
            .iter()
            .enumerate()
            .skip(possessive_index + 1)
            .take(4)
            .find_map(|(index, _)| {
                multilingual_is_nominal_head_at(profile, words, index).then_some(index)
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

fn push_multilingual_noun_compound_links(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    if !profile.allow_noun_compounds {
        return;
    }
    for (index, window) in words.windows(2).enumerate() {
        let left = window[0].as_str();
        let right = window[1].as_str();
        if multilingual_is_nominal_head_at(profile, words, index)
            && multilingual_is_nominal_head_at(profile, words, index + 1)
            && !multilingual_is_proper_name(profile, left)
            && !multilingual_is_proper_name(profile, right)
            && !multilingual_is_likely_verb(
                profile,
                left,
                index
                    .checked_sub(1)
                    .and_then(|i| words.get(i))
                    .map(String::as_str),
            )
            && !multilingual_is_likely_verb(profile, right, Some(left))
        {
            push_link(
                links,
                link(index, index + 1, SyntacticLinkKind::NounCompound, 0.73),
            );
        }
    }
}

fn push_multilingual_apposition_links(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    for (index, window) in words.windows(2).enumerate() {
        let left = window[0].as_str();
        let right = window[1].as_str();
        if (contains(left, profile.common_appositive_heads)
            && multilingual_is_proper_name(profile, right))
            || (multilingual_is_proper_name(profile, left)
                && contains(right, profile.common_appositive_heads))
        {
            push_link(
                links,
                link(index, index + 1, SyntacticLinkKind::Apposition, 0.7),
            );
        }
    }
}

fn push_multilingual_vocative_links(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    for (index, window) in words.windows(2).enumerate() {
        if contains(&window[0], profile.vocative_openers)
            && multilingual_is_nominal(profile, &window[1])
        {
            push_link(
                links,
                link(index, index + 1, SyntacticLinkKind::Vocative, 0.82),
            );
        }
    }
}

fn push_multilingual_parenthetical_links(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    for (index, window) in words.windows(2).enumerate() {
        if contains(&window[0], profile.parenthetical_markers)
            || contains(&window[1], profile.parenthetical_markers)
        {
            push_link(
                links,
                link(index, index + 1, SyntacticLinkKind::Parenthetical, 0.58),
            );
        }
    }
}

fn push_multilingual_particle_links(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    for particle_index in 1..words.len() {
        if !contains(&words[particle_index], profile.phrasal_particles) {
            continue;
        }
        if let Some(verb_index) = (0..particle_index).rev().take(2).find(|index| {
            let previous = index
                .checked_sub(1)
                .and_then(|previous| words.get(previous))
                .map(String::as_str);
            multilingual_is_likely_verb(profile, &words[*index], previous)
        }) {
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

fn push_multilingual_passive_links(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    for participle_index in 1..words.len() {
        if !is_past_participle(profile, &words[participle_index]) {
            continue;
        }
        let Some(auxiliary_index) = (0..participle_index)
            .rev()
            .take(3)
            .find(|index| multilingual_is_copula(profile, &words[*index]))
        else {
            continue;
        };
        if let Some(subject_index) = (0..auxiliary_index)
            .rev()
            .take(5)
            .find(|index| multilingual_is_subject(profile, &words[*index]))
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

fn push_multilingual_contrast_links(
    words: &[String],
    profile: GrammarRuleSet,
    links: &mut Vec<SyntacticLink>,
) {
    for (negator_index, word) in words.iter().enumerate() {
        if !contains(word, profile.contrast_negators) {
            continue;
        }
        if let Some(conjunction_index) = words
            .iter()
            .enumerate()
            .skip(negator_index + 1)
            .find_map(|(index, word)| multilingual_is_conjunction(profile, word).then_some(index))
        {
            push_link(
                links,
                link(
                    negator_index,
                    conjunction_index,
                    SyntacticLinkKind::ContrastPair,
                    0.86,
                ),
            );
        }
    }
}

fn push_multilingual_coordination_links(
    words: &[String],
    profile: GrammarRuleSet,
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
    profile: GrammarRuleSet,
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

fn multilingual_pos(profile: GrammarRuleSet, word: &str, previous: Option<&str>) -> PartOfSpeech {
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
    } else if multilingual_is_proper_name(profile, word) {
        PartOfSpeech::ProperName
    } else if multilingual_is_nominal(profile, word) {
        PartOfSpeech::Noun
    } else {
        PartOfSpeech::Unknown
    }
}

fn multilingual_pos_at(
    profile: GrammarRuleSet,
    words: &[String],
    word_index: usize,
    links: &[SyntacticLink],
) -> PartOfSpeech {
    let previous = word_index
        .checked_sub(1)
        .and_then(|index| words.get(index))
        .map(String::as_str);
    let base = multilingual_pos(profile, &words[word_index], previous);
    let has_incoming = |kind| {
        links
            .iter()
            .any(|link| link.right == word_index && link.kind == kind)
    };
    match base {
        PartOfSpeech::Noun | PartOfSpeech::Unknown
            if has_incoming(SyntacticLinkKind::Auxiliary) =>
        {
            PartOfSpeech::Verb
        }
        PartOfSpeech::Verb if has_incoming(SyntacticLinkKind::Determiner) => PartOfSpeech::Noun,
        PartOfSpeech::Unknown
            if has_incoming(SyntacticLinkKind::Determiner)
                || multilingual_is_nominal_head_at(profile, words, word_index) =>
        {
            PartOfSpeech::Noun
        }
        _ => base,
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

    pub fn raw_grammar_parses(&self) -> &[RawGrammarParse] {
        &self.raw_link_grammar_parses
    }

    pub fn raw_grammar_parses_mut(&mut self) -> &mut Vec<RawGrammarParse> {
        &mut self.raw_link_grammar_parses
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

pub(crate) fn normalize_syntax_word(word: &str) -> String {
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

pub(crate) fn is_syntax_word_character(character: char) -> bool {
    character.is_alphabetic()
        || character == '\''
        || matches!(character, '\u{0300}'..='\u{036F}' | '\u{0900}'..='\u{094D}')
}

pub(crate) fn link(
    left: usize,
    right: usize,
    kind: SyntacticLinkKind,
    confidence: f32,
) -> SyntacticLink {
    SyntacticLink {
        left,
        right,
        kind,
        confidence,
        source: SyntacticLinkSource::GrammarRule,
    }
}

pub(crate) fn push_link(links: &mut Vec<SyntacticLink>, link: SyntacticLink) {
    if !links.iter().any(|existing| {
        existing.left == link.left && existing.right == link.right && existing.kind == link.kind
    }) {
        links.push(link);
    }
}

fn multilingual_is_determiner(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.determiners)
}

fn multilingual_is_pronoun(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.pronouns)
}

fn multilingual_is_auxiliary(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.auxiliaries)
}

fn multilingual_is_copula(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.copulas)
}

fn multilingual_is_preposition(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.prepositions)
}

fn multilingual_is_postposition(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.postpositions)
}

fn multilingual_is_conjunction(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.conjunctions) || has_enclitic_suffix(word, profile.enclitic_suffixes)
}

fn multilingual_is_particle(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.particles) || has_enclitic_suffix(word, profile.enclitic_suffixes)
}

fn multilingual_is_complementizer(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.complementizers)
}

fn multilingual_is_complementizer_at(
    profile: GrammarRuleSet,
    words: &[String],
    index: usize,
) -> bool {
    if !multilingual_is_complementizer(profile, &words[index]) {
        return false;
    }
    if multilingual_is_determiner(profile, &words[index])
        && words.get(index + 1).is_some_and(|next| {
            multilingual_is_nominal_at(profile, words, index + 1)
                && !multilingual_is_pronoun(profile, next)
                && !multilingual_is_verbal_lexeme(profile, next)
        })
    {
        return false;
    }
    true
}

fn multilingual_is_relative_marker(profile: GrammarRuleSet, word: &str) -> bool {
    multilingual_is_complementizer(profile, word)
}

fn multilingual_is_relative_marker_at(
    profile: GrammarRuleSet,
    words: &[String],
    index: usize,
) -> bool {
    multilingual_is_complementizer_at(profile, words, index)
}

fn multilingual_is_object_pronoun(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.object_pronouns)
}

fn multilingual_is_adverb(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.adverbs) || has_suffix(word, profile.adverb_suffixes)
}

fn multilingual_is_adjective(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.adjectives) || has_suffix(word, profile.adjective_suffixes)
}

fn multilingual_is_likely_verb(
    profile: GrammarRuleSet,
    word: &str,
    previous: Option<&str>,
) -> bool {
    if multilingual_is_auxiliary(profile, word) {
        return true;
    }
    if previous.is_some_and(|previous| {
        multilingual_is_determiner(profile, previous)
            && !multilingual_is_object_pronoun(profile, previous)
            && !multilingual_is_complementizer(profile, previous)
    }) {
        return false;
    }
    if multilingual_is_pronoun(profile, word)
        || multilingual_is_determiner(profile, word)
        || multilingual_is_preposition(profile, word)
        || multilingual_is_conjunction(profile, word)
        || multilingual_is_complementizer(profile, word)
    {
        return false;
    }
    if contains(word, profile.non_verbs) {
        return false;
    }
    multilingual_is_verbal_lexeme(profile, word)
        || (previous.is_some_and(|previous| multilingual_is_subject_pronoun(profile, previous))
            && has_suffix(word, profile.subject_verb_suffixes))
}

fn multilingual_is_verbal_lexeme(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.verbs) || has_suffix(word, profile.verb_suffixes)
}

fn multilingual_is_nominal(profile: GrammarRuleSet, word: &str) -> bool {
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

fn multilingual_is_nominal_at(profile: GrammarRuleSet, words: &[String], index: usize) -> bool {
    if multilingual_is_nominal(profile, &words[index]) {
        return true;
    }
    let previous = index
        .checked_sub(1)
        .and_then(|previous| words.get(previous))
        .map(String::as_str);
    previous.is_some_and(|previous| multilingual_is_determiner(profile, previous))
        && !multilingual_is_determiner(profile, &words[index])
        && !multilingual_is_pronoun(profile, &words[index])
        && !multilingual_is_preposition(profile, &words[index])
        && !multilingual_is_conjunction(profile, &words[index])
        && !multilingual_is_complementizer(profile, &words[index])
}

fn multilingual_is_nominal_head(profile: GrammarRuleSet, word: &str) -> bool {
    multilingual_is_nominal(profile, word)
        && !multilingual_is_determiner(profile, word)
        && !multilingual_is_adjective(profile, word)
        && !multilingual_is_pronoun(profile, word)
}

fn multilingual_is_nominal_head_at(
    profile: GrammarRuleSet,
    words: &[String],
    index: usize,
) -> bool {
    multilingual_is_nominal_head(profile, &words[index])
        || (multilingual_is_nominal_at(profile, words, index)
            && !multilingual_is_determiner(profile, &words[index])
            && !multilingual_is_adjective(profile, &words[index])
            && !multilingual_is_pronoun(profile, &words[index]))
}

fn multilingual_is_object_candidate(profile: GrammarRuleSet, word: &str) -> bool {
    multilingual_is_object_pronoun(profile, word)
        || has_suffix(word, profile.object_suffixes)
        || multilingual_is_nominal_head(profile, word)
}

fn multilingual_is_subject(profile: GrammarRuleSet, word: &str) -> bool {
    multilingual_is_subject_pronoun(profile, word)
        || (has_suffix(word, profile.subject_suffixes)
            && !has_suffix(word, profile.object_suffixes))
        || (multilingual_is_nominal_head(profile, word)
            && !has_suffix(word, profile.object_suffixes)
            && !multilingual_is_object_pronoun(profile, word))
}

fn multilingual_is_connector_subject(profile: GrammarRuleSet, word: &str) -> bool {
    multilingual_is_subject_pronoun(profile, word)
        || (has_suffix(word, profile.subject_suffixes)
            && !has_suffix(word, profile.object_suffixes))
        || multilingual_is_proper_name(profile, word)
}

fn multilingual_is_subject_pronoun(profile: GrammarRuleSet, word: &str) -> bool {
    multilingual_is_pronoun(profile, word)
        && !multilingual_is_object_pronoun(profile, word)
        && !multilingual_is_complementizer(profile, word)
}

fn multilingual_is_proper_name(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.proper_names)
}

fn is_possessive_nominal(profile: GrammarRuleSet, word: &str) -> bool {
    profile.possessive_suffixes.iter().any(|suffix| {
        word.strip_suffix(suffix)
            .is_some_and(|stem| !stem.is_empty())
    })
}

fn is_past_participle(profile: GrammarRuleSet, word: &str) -> bool {
    contains(word, profile.past_participles)
        || profile
            .past_participles
            .iter()
            .any(|participle| *participle == "*" && word.ends_with("ed"))
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
        VarietyGrammarParser::new(VarietyId(code.into())).parse(&words(sentence), None)
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
    fn projects_udpipe_conllu_into_uniform_grammar_analysis() {
        let words = ["they", "look", "at", "us"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let conllu = "\
1\tthey\t_\tPRON\t_\t_\t2\tnsubj\t_\t_
2\tlook\t_\tVERB\t_\t_\t0\troot\t_\t_
3\tat\t_\tADP\t_\t_\t4\tcase\t_\t_
4\tus\t_\tPRON\t_\t_\t2\tobl\t_\t_
";
        let analysis =
            analysis_from_udpipe_conllu(&words, Some(TerminalPunctuation::Period), conllu)
                .expect("fixture should project");

        assert_eq!(analysis.tokens[1].pos, PartOfSpeech::Verb);
        assert_eq!(
            analysis
                .raw_link_grammar_parses
                .first()
                .map(|parse| parse.backend),
            Some(RawLinkGrammarBackend::UdPipe)
        );
        assert_link_between(&analysis, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&analysis, 2, 3, SyntacticLinkKind::Preposition);
    }

    #[test]
    fn parses_auxiliary_and_coordination_links() {
        let words = ["do", "you", "want", "either", "tea", "or", "coffee"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let analysis = VarietyGrammarParser::new(VarietyId("en-US-GA".into()))
            .parse(&words, Some(TerminalPunctuation::Question));

        assert!(analysis.word_has_link(0, SyntacticLinkKind::Auxiliary));
        assert!(analysis.word_has_link(5, SyntacticLinkKind::Coordination));
        assert!(analysis.matches_environment_pattern(&EnvironmentPattern {
            predicates: vec![ContextPredicate::SyntacticLink(
                SyntacticLinkKind::Coordination
            )],
        }));
    }

    #[test]
    fn builtin_varieties_use_shared_grammar_engine() {
        let samples = [
            ("en-US-GA", "the dog chased the cat"),
            ("fr-FR-Standard", "je vois la maison"),
            ("es-ES-Castilian", "yo veo la casa"),
            ("de-DE-Standard", "ich sehe das buch"),
            ("eo", "mi vidas la libron"),
            ("la-Classical", "puella puerum amat"),
            ("el-GR-Standard", "εγώ βλέπω τον κόσμο"),
            ("sa-Deva-Standard", "अहं फलम् खादति"),
        ];

        for (code, sentence) in samples {
            let variety = variety_by_code(code).expect("builtin variety should load");
            assert!(
                variety.syntax_analyzer.is_none(),
                "{code} should use shared grammar rules, not a parser callback"
            );
            assert!(
                variety.syntax_rules.is_some(),
                "{code} should own grammar rules"
            );

            let analysis = parse_variety(code, sentence);
            assert_eq!(
                analysis
                    .raw_link_grammar_parses
                    .first()
                    .map(|parse| parse.backend),
                Some(RawLinkGrammarBackend::TonguesRuleGrammar),
                "{code} should report the shared in-tree backend"
            );
            assert!(
                analysis
                    .primary_parse()
                    .is_some_and(|parse| !parse.links.is_empty()),
                "{code} should produce at least one typed syntax link for {sentence:?}"
            );
        }
    }

    #[test]
    fn english_connector_families_emit_typed_links() {
        // Connector families used by the in-tree English rules:
        // D, A/AN, J/MV, S/O, TO, C, and CO.
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
            let analysis = parse_variety("en-US-GA", sentence);
            for expected_link in expected_links {
                assert_link(&analysis, expected_link);
            }
        }
    }

    #[test]
    fn english_corpus_basic_samples_cover_nominal_and_clause_rules() {
        // Fixture-style coverage for nominal and clause patterns; this is not
        // delegated to an external parser installation.
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
            let analysis = parse_variety("en-US-GA", sentence);
            for expected_link in expected_links {
                assert_link(&analysis, expected_link);
            }
        }
    }

    #[test]
    fn ambiguous_verb_lexemes_emit_clause_links() {
        // Ambiguous noun/verb examples exercise local POS disambiguation enough
        // to preserve clause structure for downstream prosody rules.
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
            let analysis = parse_variety("en-US-GA", sentence);
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
        let relative = parse_variety("en-US-GA", "the man who saw mary left");
        assert_link_between(&relative, 1, 2, SyntacticLinkKind::Apposition);
        assert_link_between(&relative, 2, 3, SyntacticLinkKind::Complement);
        assert_link_between(&relative, 2, 3, SyntacticLinkKind::Subject);
        assert_link_between(&relative, 3, 4, SyntacticLinkKind::Object);

        let subordinate = parse_variety("en-US-GA", "because she left john waited");
        assert_link_between(&subordinate, 0, 2, SyntacticLinkKind::Complement);
        assert_link_between(&subordinate, 1, 2, SyntacticLinkKind::Subject);
        assert_link_between(&subordinate, 3, 4, SyntacticLinkKind::Subject);
    }

    #[test]
    fn english_parser_does_not_promote_clause_subjects_to_matrix_objects() {
        let analysis = parse_variety("en-US-GA", "i know that she left");

        assert_link_between(&analysis, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&analysis, 1, 2, SyntacticLinkKind::Complement);
        assert_link_between(&analysis, 2, 4, SyntacticLinkKind::Complement);
        assert_link_between(&analysis, 3, 4, SyntacticLinkKind::Subject);
        assert_no_link_between(&analysis, 1, 3, SyntacticLinkKind::Object);
    }

    #[test]
    fn english_parser_handles_possessives_particles_and_passives() {
        let possessive = parse_variety("en-US-GA", "mary's old friend arrived");
        assert_link_between(&possessive, 0, 2, SyntacticLinkKind::Determiner);
        assert_link_between(&possessive, 1, 2, SyntacticLinkKind::Modifier);
        assert_link_between(&possessive, 2, 3, SyntacticLinkKind::Subject);

        let particle = parse_variety("en-US-GA", "they turn off the light");
        assert_link_between(&particle, 0, 1, SyntacticLinkKind::Subject);
        assert_link_between(&particle, 1, 2, SyntacticLinkKind::Modifier);
        assert_link_between(&particle, 1, 4, SyntacticLinkKind::Object);

        let passive = parse_variety("en-US-GA", "the ball was thrown by mary");
        assert_link_between(&passive, 1, 3, SyntacticLinkKind::Subject);
        assert_link_between(&passive, 2, 3, SyntacticLinkKind::Auxiliary);
        assert_link_between(&passive, 4, 5, SyntacticLinkKind::Preposition);
    }

    #[test]
    fn english_parser_normalizes_internal_punctuation_and_skips_empty_tokens() {
        let analysis = VarietyGrammarParser::new(VarietyId("en-US-GA".into())).parse(
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
        let punctuated = VarietyGrammarParser::new(VarietyId("fr-FR-Standard".into())).parse(
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
    fn emits_vocative_from_oh_voc_pattern() {
        let analysis = parse_variety("en-US-GA", "Oh Joe listen");

        assert_link(&analysis, SyntacticLinkKind::Vocative);
    }
}
