use crate::syntax::PartOfSpeech;

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
