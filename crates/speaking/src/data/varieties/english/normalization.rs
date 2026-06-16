use crate::syntax::PartOfSpeech;

#[derive(Debug, Clone, Copy)]
pub struct SpokenFormRewrite {
    pub from: &'static str,
    pub to: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct UnitRewrite {
    pub aliases: &'static [&'static str],
    pub singular: &'static str,
    pub plural: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct PosSensitivePronunciationSpec {
    pub word: &'static str,
    pub part_of_speech: PartOfSpeech,
    pub symbols: &'static [&'static str],
}

pub const SCALE_WORDS: &[&str] = &[
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
];

const POWER_PREFIXES: &[&str] = &[
    "",
    "m",
    "b",
    "tr",
    "quadr",
    "quint",
    "sext",
    "sept",
    "oct",
    "non",
    "dec",
    "undec",
    "duodec",
    "tredec",
    "quattuordec",
    "quindec",
    "sexdec",
    "septendec",
    "octodec",
    "novemdec",
    "vigint",
];

pub const UNITS: &[UnitRewrite] = &[
    unit(&["ft", "feet", "foot"], "foot", "feet"),
    unit(&["in", "inch", "inches"], "inch", "inches"),
    unit(&["mph"], "miles per hour", "miles per hour"),
    unit(&["lb", "lbs", "pound", "pounds"], "pound", "pounds"),
    unit(
        &["kg", "kilo", "kilos", "kilograms"],
        "kilogram",
        "kilograms",
    ),
    unit(
        &["mg", "milligram", "milligrams"],
        "milligram",
        "milligrams",
    ),
    unit(&["ghz", "gigahertz"], "gigahertz", "gigahertz"),
    unit(&["fahrenheit"], "degrees Fahrenheit", "degrees Fahrenheit"),
    unit(&["celsius"], "degrees Celsius", "degrees Celsius"),
    unit(
        &["cm", "centimeter", "centimeters"],
        "centimeter",
        "centimeters",
    ),
    unit(&["m", "meter", "meters"], "meter", "meters"),
    unit(&["%", "percent"], "percent", "percent"),
];

pub const SPOKEN_FORM_REWRITES: &[SpokenFormRewrite] = &[
    rewrite("Dr.", "Doctor"),
    rewrite("Mr.", "Mister"),
    rewrite("Mrs.", "Missus"),
    rewrite("Ms.", "Miz"),
    rewrite("Rep.", "Representative"),
    rewrite("Sen.", "Senator"),
    rewrite("Gov.", "Governor"),
    rewrite("Prof.", "Professor"),
    rewrite("Sr.", "Senior"),
    rewrite("Jr.", "Junior"),
    rewrite("e.g.", "e g"),
    rewrite("E.g.", "e g"),
    rewrite("i.e.", "i e"),
    rewrite("I.e.", "i e"),
    rewrite("a.m.", "a m"),
    rewrite("A.M.", "a m"),
    rewrite("p.m.", "p m"),
    rewrite("P.M.", "p m"),
    rewrite("Jan.", "January"),
    rewrite("Feb.", "February"),
    rewrite("Mar.", "March"),
    rewrite("Apr.", "April"),
    rewrite("Jun.", "June"),
    rewrite("Jul.", "July"),
    rewrite("Aug.", "August"),
    rewrite("Sep.", "September"),
    rewrite("Sept.", "September"),
    rewrite("Oct.", "October"),
    rewrite("Nov.", "November"),
    rewrite("Dec.", "December"),
    rewrite("°F", " Fahrenheit"),
    rewrite("°C", " Celsius"),
    rewrite("AT&T", "A T and T"),
    rewrite("R&D", "R and D"),
    rewrite("C++", "C plus plus"),
    rewrite("C#", "C sharp"),
    rewrite("Loadstone", "Lodestone"),
    rewrite("loadstone", "lodestone"),
    rewrite("D.", "D"),
    rewrite("R.", "R"),
    rewrite("NY.", "New York"),
    rewrite("N.Y.", "New York"),
];

pub const POS_SENSITIVE_PRONUNCIATIONS: &[PosSensitivePronunciationSpec] = &[
    pos("close", PartOfSpeech::Adjective, &["K", "L", "OW1", "S"]),
    pos("close", PartOfSpeech::Verb, &["K", "L", "OW1", "Z"]),
    pos(
        "conduct",
        PartOfSpeech::Noun,
        &["K", "AA1", "N", "D", "AH0", "K", "T"],
    ),
    pos(
        "conduct",
        PartOfSpeech::Verb,
        &["K", "AA0", "N", "D", "AH1", "K", "T"],
    ),
    pos(
        "console",
        PartOfSpeech::Noun,
        &["K", "AA1", "N", "S", "OW0", "L"],
    ),
    pos(
        "console",
        PartOfSpeech::Verb,
        &["K", "AH0", "N", "S", "OW1", "L"],
    ),
    pos(
        "object",
        PartOfSpeech::Noun,
        &["AA1", "B", "JH", "EH0", "K", "T"],
    ),
    pos(
        "object",
        PartOfSpeech::Verb,
        &["AH0", "B", "JH", "EH1", "K", "T"],
    ),
    pos("permit", PartOfSpeech::Noun, &["P", "ER1", "M", "IH2", "T"]),
    pos("permit", PartOfSpeech::Verb, &["P", "ER0", "M", "IH1", "T"]),
    pos(
        "present",
        PartOfSpeech::Adjective,
        &["P", "R", "EH1", "Z", "AH0", "N", "T"],
    ),
    pos(
        "present",
        PartOfSpeech::Noun,
        &["P", "R", "EH1", "Z", "AH0", "N", "T"],
    ),
    pos(
        "present",
        PartOfSpeech::Verb,
        &["P", "R", "IY0", "Z", "EH1", "N", "T"],
    ),
    pos(
        "produce",
        PartOfSpeech::Noun,
        &["P", "R", "OW1", "D", "UW0", "S"],
    ),
    pos(
        "produce",
        PartOfSpeech::Verb,
        &["P", "R", "AH0", "D", "UW1", "S"],
    ),
    pos(
        "project",
        PartOfSpeech::Noun,
        &["P", "R", "AA1", "JH", "EH0", "K", "T"],
    ),
    pos(
        "project",
        PartOfSpeech::Verb,
        &["P", "R", "AH0", "JH", "EH1", "K", "T"],
    ),
    pos("rebel", PartOfSpeech::Noun, &["R", "EH1", "B", "AH0", "L"]),
    pos("rebel", PartOfSpeech::Verb, &["R", "IH0", "B", "EH1", "L"]),
    pos("record", PartOfSpeech::Noun, &["R", "EH1", "K", "ER0", "D"]),
    pos(
        "record",
        PartOfSpeech::Verb,
        &["R", "AH0", "K", "AO1", "R", "D"],
    ),
    pos(
        "refuse",
        PartOfSpeech::Noun,
        &["R", "EH1", "F", "Y", "UW2", "Z"],
    ),
    pos(
        "refuse",
        PartOfSpeech::Verb,
        &["R", "AH0", "F", "Y", "UW1", "Z"],
    ),
    pos(
        "subject",
        PartOfSpeech::Noun,
        &["S", "AH1", "B", "JH", "IH0", "K", "T"],
    ),
    pos(
        "subject",
        PartOfSpeech::Verb,
        &["S", "AH0", "B", "JH", "EH1", "K", "T"],
    ),
    pos("wind", PartOfSpeech::Noun, &["W", "IH1", "N", "D"]),
    pos("wind", PartOfSpeech::Verb, &["W", "AY1", "N", "D"]),
    pos("lead", PartOfSpeech::Noun, &["L", "EH1", "D"]),
    pos("lead", PartOfSpeech::Adjective, &["L", "EH1", "D"]),
    pos("lead", PartOfSpeech::Verb, &["L", "IY1", "D"]),
];

pub fn is_scale_word(word: &str) -> bool {
    SCALE_WORDS.contains(&word)
}

pub fn unit_spoken_form(value: u128, unit_alias: &str) -> Option<&'static str> {
    let unit = UNITS
        .iter()
        .find(|unit| unit.aliases.contains(&unit_alias))?;
    Some(if value == 1 {
        unit.singular
    } else {
        unit.plural
    })
}

pub fn is_known_unit(word: &str) -> bool {
    unit_spoken_form(2, word).is_some()
}

pub fn spell_cardinal(value: u128) -> String {
    match value {
        0 => "zero".to_string(),
        1 => "one".to_string(),
        2 => "two".to_string(),
        3 => "three".to_string(),
        4 => "four".to_string(),
        5 => "five".to_string(),
        6 => "six".to_string(),
        7 => "seven".to_string(),
        8 => "eight".to_string(),
        9 => "nine".to_string(),
        10 => "ten".to_string(),
        11 => "eleven".to_string(),
        12 => "twelve".to_string(),
        13 => "thirteen".to_string(),
        14 => "fourteen".to_string(),
        15 => "fifteen".to_string(),
        16 => "sixteen".to_string(),
        17 => "seventeen".to_string(),
        18 => "eighteen".to_string(),
        19 => "nineteen".to_string(),
        20 => "twenty".to_string(),
        30 => "thirty".to_string(),
        40 => "forty".to_string(),
        50 => "fifty".to_string(),
        60 => "sixty".to_string(),
        70 => "seventy".to_string(),
        80 => "eighty".to_string(),
        90 => "ninety".to_string(),
        _ => {
            let log10 = ilog10_u128(value);
            match log10 {
                1 => {
                    let head = value / 10;
                    let tail = value % 10;
                    format!("{}-{}", spell_cardinal(head * 10), spell_cardinal(tail))
                }
                2 => {
                    let head = value / 100;
                    let tail = value % 100;
                    if tail > 0 {
                        format!("{} hundred {}", spell_cardinal(head), spell_cardinal(tail))
                    } else {
                        format!("{} hundred", spell_cardinal(head))
                    }
                }
                _ => {
                    let power = (log10 / 3) - 1;
                    let num_digits = log10 - (log10 % 3);
                    let divisor = 10u128.pow(num_digits);
                    let head = value / divisor;
                    let tail = value % divisor;

                    let Ok(power_name) = power_name(power as usize) else {
                        return value.to_string();
                    };

                    if tail > 0 {
                        format!(
                            "{} {} {}",
                            spell_cardinal(head),
                            power_name,
                            spell_cardinal(tail)
                        )
                    } else {
                        format!("{} {}", spell_cardinal(head), power_name)
                    }
                }
            }
        }
    }
}

pub fn spell_ordinal(value: u128) -> String {
    match value {
        0 => "zeroth".to_string(),
        1 => "first".to_string(),
        2 => "second".to_string(),
        3 => "third".to_string(),
        4 => "fourth".to_string(),
        5 => "fifth".to_string(),
        6 => "sixth".to_string(),
        7 => "seventh".to_string(),
        8 => "eighth".to_string(),
        9 => "ninth".to_string(),
        10 => "tenth".to_string(),
        11 => "eleventh".to_string(),
        12 => "twelfth".to_string(),
        13 => "thirteenth".to_string(),
        14 => "fourteenth".to_string(),
        15 => "fifteenth".to_string(),
        16 => "sixteenth".to_string(),
        17 => "seventeenth".to_string(),
        18 => "eighteenth".to_string(),
        19 => "nineteenth".to_string(),
        20 => "twentieth".to_string(),
        30 => "thirtieth".to_string(),
        40 => "fortieth".to_string(),
        50 => "fiftieth".to_string(),
        60 => "sixtieth".to_string(),
        70 => "seventieth".to_string(),
        80 => "eightieth".to_string(),
        90 => "ninetieth".to_string(),
        _ if value < 100 && value % 10 != 0 => {
            format!(
                "{}-{}",
                spell_cardinal(value - (value % 10)),
                spell_ordinal(value % 10)
            )
        }
        _ => format!("{}th", spell_cardinal(value)),
    }
}

pub fn spell_year(year: u128) -> String {
    if (2000..=2009).contains(&year) {
        format!("two thousand {}", spell_cardinal(year - 2000))
    } else if (2010..=2099).contains(&year) {
        format!("twenty {}", spell_cardinal(year - 2000))
    } else {
        spell_cardinal(year)
    }
}

fn power_name(power: usize) -> Result<String, &'static str> {
    if power == 0 {
        return Ok("thousand".to_string());
    }
    if power == 100 {
        return Ok("centillion".to_string());
    }
    if power >= POWER_PREFIXES.len() {
        return Err("number is too large to spell");
    }
    Ok(format!("{}illion", POWER_PREFIXES[power]))
}

fn ilog10_u128(mut value: u128) -> u32 {
    let mut count = 0;
    while value >= 10 {
        value /= 10;
        count += 1;
    }
    count
}

const fn rewrite(from: &'static str, to: &'static str) -> SpokenFormRewrite {
    SpokenFormRewrite { from, to }
}

const fn unit(
    aliases: &'static [&'static str],
    singular: &'static str,
    plural: &'static str,
) -> UnitRewrite {
    UnitRewrite {
        aliases,
        singular,
        plural,
    }
}

const fn pos(
    word: &'static str,
    part_of_speech: PartOfSpeech,
    symbols: &'static [&'static str],
) -> PosSensitivePronunciationSpec {
    PosSensitivePronunciationSpec {
        word,
        part_of_speech,
        symbols,
    }
}
