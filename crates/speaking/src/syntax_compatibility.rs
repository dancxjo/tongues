//! Deprecated source and wire names retained during the v1 terminology migration.
//!
//! New code must use the backend-neutral types in [`crate::syntax`]. These aliases
//! are scheduled for removal on 2027-07-27.

use crate::segment::TerminalPunctuation;
use crate::syntax::{
    BackendCost, BackendLink, BackendParse, GrammarAnalysis, GrammarBackend, GrammarConnector,
    GrammarParser, GrammarRule, GrammarRuleSet, RankedGrammarParse, VarietyGrammarParser,
};

#[deprecated(since = "0.1.0", note = "use GrammarAnalysis; removal 2027-07-27")]
pub type SentenceSyntaxAnalysis = GrammarAnalysis;
#[deprecated(since = "0.1.0", note = "use RankedGrammarParse; removal 2027-07-27")]
pub type SyntacticLinkParse = RankedGrammarParse;
#[deprecated(since = "0.1.0", note = "use BackendParse; removal 2027-07-27")]
pub type RawLinkGrammarParse = BackendParse;
#[deprecated(since = "0.1.0", note = "use BackendLink; removal 2027-07-27")]
pub type RawLinkGrammarLink = BackendLink;
#[deprecated(since = "0.1.0", note = "use BackendCost; removal 2027-07-27")]
pub type RawLinkGrammarCost = BackendCost;
#[deprecated(since = "0.1.0", note = "use GrammarBackend; removal 2027-07-27")]
pub type RawLinkGrammarBackend = GrammarBackend;
#[deprecated(since = "0.1.0", note = "use VarietyGrammarParser; removal 2027-07-27")]
pub type VarietyLinkGrammarParser = VarietyGrammarParser;
#[deprecated(since = "0.1.0", note = "use GrammarRuleSet; removal 2027-07-27")]
pub type LinkGrammarRuleSet = GrammarRuleSet;
#[deprecated(since = "0.1.0", note = "use GrammarRule; removal 2027-07-27")]
pub type LinkGrammarRule = GrammarRule;
#[deprecated(since = "0.1.0", note = "use GrammarConnector; removal 2027-07-27")]
pub type LinkGrammarConnector = GrammarConnector;

#[deprecated(since = "0.1.0", note = "use GrammarParser; removal 2027-07-27")]
pub trait LinkGrammarParser: GrammarParser {}

#[allow(deprecated)]
impl<T: GrammarParser + ?Sized> LinkGrammarParser for T {}

#[deprecated(
    since = "0.1.0",
    note = "use parse_grammar_with_rules; removal 2027-07-27"
)]
pub fn parse_link_grammar_with_rules(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
    profile: GrammarRuleSet,
) -> GrammarAnalysis {
    crate::syntax::parse_grammar_with_rules(words, terminal, profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn deprecated_source_aliases_still_compile() {
        let _: SentenceSyntaxAnalysis = GrammarAnalysis::default();
        let _: Option<RawLinkGrammarParse> = None;
        let _: Option<VarietyLinkGrammarParser> = None;
    }
}
