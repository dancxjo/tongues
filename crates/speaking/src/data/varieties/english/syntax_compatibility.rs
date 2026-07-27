//! Deprecated English parser entry points retained until 2027-07-27.

use crate::segment::TerminalPunctuation;
use crate::syntax::GrammarAnalysis;

#[deprecated(since = "0.1.0", note = "use parse_grammar; removal 2027-07-27")]
pub fn parse_link_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
) -> GrammarAnalysis {
    super::syntax::parse_grammar(words, terminal)
}

#[deprecated(
    since = "0.1.0",
    note = "use parse_english_grammar; removal 2027-07-27"
)]
pub fn parse_english_link_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
) -> GrammarAnalysis {
    super::syntax::parse_english_grammar(words, terminal)
}
