//! Unified vocabulary for tongues sequence translation.
//!
//! Provides a single character-level `Vocab` that maps orthographic forms,
//! phonemic/phonetic forms, and task/control tokens to shared IDs.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Special control token IDs ──────────────────────────────────────────────

pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const EOS_ID: u32 = 3;
pub const SEP_ID: u32 = 4;

// ── Task prefix token IDs ──────────────────────────────────────────────────

pub const G2P_ID: u32 = 5;
pub const P2G_ID: u32 = 6;

pub const SPECIAL_COUNT: u32 = 7;

// ── Unified Vocab ──────────────────────────────────────────────────────────

/// Bidirectional map between characters/special tokens and integer IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    /// Index → token string (position == ID).
    pub tokens: Vec<String>,
    /// Token string → ID.
    pub token_to_id: HashMap<String, u32>,
}

impl Vocab {
    /// Construct a unified vocabulary from words, phonemes, and phones.
    pub fn build(words: &[String], phonemes: &[String], phones: &[String]) -> Self {
        let mut tokens: Vec<String> = vec![
            "<PAD>".into(),
            "<UNK>".into(),
            "<BOS>".into(),
            "<EOS>".into(),
            "<SEP>".into(),
            "<G2P>".into(),
            "<P2G>".into(),
            "<task:orthography_to_phonology>".into(),
            "<task:phonology_to_orthography>".into(),
            "<task:phonetic_realization>".into(),
            "<task:align>".into(),
            "<task:normalize>".into(),
            "<task:guess_lang_from_orthography>".into(),
            "<task:guess_lang_from_phonology>".into(),
            "<task:guess_lang_from_orthography_and_phonology>".into(),
        ];

        let mut control_tokens = std::collections::BTreeSet::new();
        let mut seen = std::collections::BTreeSet::new();
        seed_broad_linguistic_vocab(&mut control_tokens, &mut seen);

        // Collect all unique characters
        for word in words {
            collect_angle_bracket_tokens(word, &mut control_tokens);
            for c in word.chars() {
                seen.insert(c.to_string());
            }
        }
        for pm in phonemes {
            collect_angle_bracket_tokens(pm, &mut control_tokens);
            for c in pm.chars() {
                seen.insert(c.to_string());
            }
        }
        for ph in phones {
            collect_angle_bracket_tokens(ph, &mut control_tokens);
            for c in ph.chars() {
                seen.insert(c.to_string());
            }
        }

        for token in control_tokens {
            if !tokens.contains(&token) {
                tokens.push(token);
            }
        }
        tokens.extend(seen);

        let token_to_id: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        Vocab {
            tokens,
            token_to_id,
        }
    }

    /// Look up the ID for a token string, returning `UNK_ID` for unknown tokens.
    pub fn get_id(&self, token: &str) -> u32 {
        *self.token_to_id.get(token).unwrap_or(&UNK_ID)
    }

    /// Look up the token string for an ID.
    pub fn get_token(&self, id: u32) -> &str {
        self.tokens
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<UNK>")
    }

    /// Total number of tokens including specials.
    pub fn size(&self) -> usize {
        self.tokens.len()
    }

    /// Encode a string as IDs, preserving known `<...>` control tokens as atoms.
    pub fn encode_string(&self, s: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut index = 0;
        while index < s.len() {
            let rest = &s[index..];
            if rest.starts_with('<') {
                if let Some(end) = rest.find('>') {
                    let candidate = &rest[..=end];
                    if let Some(id) = self.token_to_id.get(candidate) {
                        ids.push(*id);
                        index += candidate.len();
                        continue;
                    }
                }
            }

            let Some(ch) = rest.chars().next() else {
                break;
            };
            ids.push(self.get_id(&ch.to_string()));
            index += ch.len_utf8();
        }
        ids
    }

    /// Decode a list of IDs back to a string (filtering out PAD/BOS/EOS/SEP).
    pub fn decode_ids(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| self.get_token(id))
            .filter(|&tok| tok != "<PAD>" && tok != "<BOS>" && tok != "<EOS>" && tok != "<SEP>")
            .collect::<Vec<_>>()
            .join("")
    }
}

fn collect_angle_bracket_tokens(value: &str, out: &mut std::collections::BTreeSet<String>) {
    let mut offset = 0;
    while let Some(start) = value[offset..].find('<') {
        let start = offset + start;
        let Some(end) = value[start..].find('>').map(|end| start + end) else {
            break;
        };
        if end > start + 1 {
            out.insert(value[start..=end].to_string());
        }
        offset = end + 1;
    }
}

fn seed_broad_linguistic_vocab(
    control_tokens: &mut std::collections::BTreeSet<String>,
    seen: &mut std::collections::BTreeSet<String>,
) {
    for token in [
        "<lang:eng>",
        "<lang:fra>",
        "<lang:deu>",
        "<lang:spa>",
        "<lang:lat>",
        "<lang:ell>",
        "<lang:grc>",
        "<lang:san>",
        "<repr:phonemes>",
        "<repr:phones>",
        "<repr:diaphonemes>",
        "<META>",
        "</META>",
        "<accent:genam>",
        "<accent:rp>",
        "<accent:ssb>",
        "<accent:aave>",
        "<accent:mle>",
        "<accent:castilian>",
        "<accent:latam>",
        "<region:us>",
        "<region:uk>",
        "<region:canada>",
        "<region:australia>",
        "<region:new_zealand>",
        "<region:ireland>",
        "<region:scotland>",
        "<region:wales>",
        "<region:south_africa>",
        "<region:nyc>",
        "<region:southern_us>",
        "<region:midland_us>",
        "<region:mid_atlantic>",
        "<feature:fronting>",
        "<feature:monophthongization>",
        "<feature:cot_caught>",
        "<feature:non_cot_caught>",
        "<feature:foot_strut_split>",
        "<feature:non_foot_strut_split>",
        "<feature:wine_whine>",
        "<feature:non_wine_whine>",
        "<feature:lot_cloth_split>",
        "<feature:ae_tensing>",
        "<feature:non_ae_tensing>",
        "<feature:ae_raising>",
        "<feature:non_ae_raising>",
        "<feature:weak_vowel>",
        "<feature:weak_form>",
        "<usage:greek_name>",
        "<usage:latin>",
        "<usage:neo_latin_scientific>",
        "<usage:legal_latin>",
        "<usage:grammatical_note>",
        "<usage:archaic>",
        "<usage:dated>",
        "<usage:obsolete>",
        "<usage:colloquial>",
        "<usage:dialectal>",
        "<usage:nonstandard>",
        "<usage:proscribed>",
        "<usage:rare>",
    ] {
        control_tokens.insert(token.to_string());
    }

    for c in " \t\n-_'’.,;:!?/[](){}<>+*=~·ˈˌ.|‿͜͡".chars() {
        seen.insert(c.to_string());
    }

    for c in 0x20_u8..=0x7e {
        seen.insert((c as char).to_string());
    }

    for (start, end) in [
        (0x00A0, 0x024F), // Latin-1, Latin Extended-A/B, IPA-adjacent letters.
        (0x0250, 0x02AF), // IPA Extensions.
        (0x02B0, 0x02FF), // Spacing Modifier Letters.
        (0x0300, 0x036F), // Combining Diacritical Marks.
        (0x0370, 0x03FF), // Greek and Coptic.
        (0x0400, 0x04FF), // Cyrillic.
        (0x0590, 0x05FF), // Hebrew.
        (0x0600, 0x06FF), // Arabic.
        (0x0700, 0x074F), // Syriac.
        (0x0750, 0x077F), // Arabic Supplement.
        (0x0780, 0x07BF), // Thaana.
        (0x07C0, 0x07FF), // NKo.
        (0x0800, 0x083F), // Samaritan.
        (0x0840, 0x085F), // Mandaic.
        (0x0860, 0x086F), // Syriac Supplement.
        (0x0870, 0x089F), // Arabic Extended-B.
        (0x08A0, 0x08FF), // Arabic Extended-A.
        (0x0900, 0x097F), // Devanagari.
        (0x0980, 0x09FF), // Bengali.
        (0x0A00, 0x0A7F), // Gurmukhi.
        (0x0A80, 0x0AFF), // Gujarati.
        (0x0B00, 0x0B7F), // Oriya.
        (0x0B80, 0x0BFF), // Tamil.
        (0x0C00, 0x0C7F), // Telugu.
        (0x0C80, 0x0CFF), // Kannada.
        (0x0D00, 0x0D7F), // Malayalam.
        (0x0D80, 0x0DFF), // Sinhala.
        (0x0E00, 0x0E7F), // Thai.
        (0x0E80, 0x0EFF), // Lao.
        (0x0F00, 0x0FFF), // Tibetan.
        (0x1000, 0x109F), // Myanmar.
        (0x10A0, 0x10FF), // Georgian.
        (0x1100, 0x11FF), // Hangul Jamo.
        (0x1200, 0x137F), // Ethiopic.
        (0x1380, 0x139F), // Ethiopic Supplement.
        (0x13A0, 0x13FF), // Cherokee.
        (0x1400, 0x167F), // Unified Canadian Aboriginal Syllabics.
        (0x1680, 0x169F), // Ogham.
        (0x16A0, 0x16FF), // Runic.
        (0x1700, 0x171F), // Tagalog.
        (0x1720, 0x173F), // Hanunoo.
        (0x1740, 0x175F), // Buhid.
        (0x1760, 0x177F), // Tagbanwa.
        (0x1780, 0x17FF), // Khmer.
        (0x1800, 0x18AF), // Mongolian.
        (0x18B0, 0x18FF), // Unified Canadian Aboriginal Syllabics Extended.
        (0x1900, 0x194F), // Limbu.
        (0x1950, 0x197F), // Tai Le.
        (0x1980, 0x19DF), // New Tai Lue.
        (0x19E0, 0x19FF), // Khmer Symbols.
        (0x1A00, 0x1A1F), // Buginese.
        (0x1A20, 0x1AAF), // Tai Tham.
        (0x1AB0, 0x1AFF), // Combining Diacritical Marks Extended.
        (0x1B00, 0x1B7F), // Balinese.
        (0x1B80, 0x1BBF), // Sundanese.
        (0x1BC0, 0x1BFF), // Batak.
        (0x1C00, 0x1C4F), // Lepcha.
        (0x1C50, 0x1C7F), // Ol Chiki.
        (0x1C80, 0x1C8F), // Cyrillic Extended-C.
        (0x1C90, 0x1CBF), // Georgian Extended.
        (0x1CC0, 0x1CCF), // Sundanese Supplement.
        (0x1CD0, 0x1CFF), // Vedic Extensions.
        (0x1D00, 0x1D7F), // Phonetic Extensions.
        (0x1D80, 0x1DBF), // Phonetic Extensions Supplement.
        (0x1DC0, 0x1DFF), // Combining Diacritical Marks Supplement.
        (0x1E00, 0x1EFF), // Latin Extended Additional.
        (0x1F00, 0x1FFF), // Greek Extended.
        (0x2000, 0x206F), // General punctuation.
        (0x2070, 0x209F), // Superscripts and Subscripts.
        (0x20A0, 0x20CF), // Currency Symbols.
        (0x20D0, 0x20FF), // Combining Diacritical Marks for Symbols.
        (0x2100, 0x214F), // Letterlike Symbols.
        (0x2150, 0x218F), // Number Forms.
        (0x2190, 0x21FF), // Arrows.
        (0x2200, 0x22FF), // Mathematical Operators.
        (0x2300, 0x23FF), // Miscellaneous Technical.
        (0x2440, 0x245F), // Optical Character Recognition.
        (0x2460, 0x24FF), // Enclosed Alphanumerics.
        (0x2500, 0x257F), // Box Drawing.
        (0x2580, 0x259F), // Block Elements.
        (0x25A0, 0x25FF), // Geometric Shapes.
        (0x2600, 0x26FF), // Miscellaneous Symbols.
        (0x2700, 0x27BF), // Dingbats.
        (0x27C0, 0x27EF), // Miscellaneous Mathematical Symbols-A.
        (0x27F0, 0x27FF), // Supplemental Arrows-A.
        (0x2800, 0x28FF), // Braille Patterns.
        (0x2900, 0x297F), // Supplemental Arrows-B.
        (0x2980, 0x29FF), // Miscellaneous Mathematical Symbols-B.
        (0x2A00, 0x2AFF), // Supplemental Mathematical Operators.
        (0x2B00, 0x2BFF), // Miscellaneous Symbols and Arrows.
        (0x2C00, 0x2C5F), // Glagolitic.
        (0x2C60, 0x2C7F), // Latin Extended-C.
        (0x2C80, 0x2CFF), // Coptic.
        (0x2D00, 0x2D2F), // Georgian Supplement.
        (0x2D30, 0x2D7F), // Tifinagh.
        (0x2D80, 0x2DDF), // Ethiopic Extended.
        (0x2DE0, 0x2DFF), // Cyrillic Extended-A.
        (0x2E00, 0x2E7F), // Supplemental Punctuation.
        (0x2E80, 0x2EFF), // CJK Radicals Supplement.
        (0x2F00, 0x2FDF), // Kangxi Radicals.
        (0x2FF0, 0x2FFF), // Ideographic Description Characters.
        (0x3000, 0x303F), // CJK Symbols and Punctuation.
        (0x3040, 0x309F), // Hiragana.
        (0x30A0, 0x30FF), // Katakana.
        (0x3100, 0x312F), // Bopomofo.
        (0x3130, 0x318F), // Hangul Compatibility Jamo.
        (0x3190, 0x319F), // Kanbun.
        (0x31A0, 0x31BF), // Bopomofo Extended.
        (0x31C0, 0x31EF), // CJK Strokes.
        (0x31F0, 0x31FF), // Katakana Phonetic Extensions.
        (0x3200, 0x32FF), // Enclosed CJK Letters and Months.
        (0x3300, 0x33FF), // CJK Compatibility.
        (0x3400, 0x4DBF), // CJK Unified Ideographs Extension A.
        (0x4DC0, 0x4DFF), // Yijing Hexagram Symbols.
        (0x4E00, 0x9FFF), // CJK Unified Ideographs.
        (0xA000, 0xA48F), // Yi Syllables.
        (0xA490, 0xA4CF), // Yi Radicals.
        (0xA4D0, 0xA4FF), // Lisu.
        (0xA500, 0xA63F), // Vai.
        (0xA640, 0xA69F), // Cyrillic Extended-B.
        (0xA6A0, 0xA6FF), // Bamum.
        (0xA700, 0xA71F), // Modifier Tone Letters.
        (0xA720, 0xA7FF), // Latin Extended-D.
        (0xA800, 0xA82F), // Syloti Nagri.
        (0xA830, 0xA83F), // Common Indic Number Forms.
        (0xA840, 0xA87F), // Phags-pa.
        (0xA880, 0xA8DF), // Saurashtra.
        (0xA8E0, 0xA8FF), // Devanagari Extended.
        (0xA900, 0xA92F), // Kayah Li.
        (0xA930, 0xA95F), // Rejang.
        (0xA960, 0xA97F), // Hangul Jamo Extended-A.
        (0xA980, 0xA9DF), // Javanese.
        (0xA9E0, 0xA9FF), // Myanmar Extended-B.
        (0xAA00, 0xAA5F), // Cham.
        (0xAA60, 0xAA7F), // Myanmar Extended-A.
        (0xAA80, 0xAADF), // Tai Viet.
        (0xAAE0, 0xAAFF), // Meetei Mayek Extensions.
        (0xAB00, 0xAB2F), // Ethiopic Extended-A.
        (0xAB30, 0xAB6F), // Latin Extended-E.
        (0xAB70, 0xABBF), // Cherokee Supplement.
        (0xABC0, 0xABFF), // Meetei Mayek.
        (0xAC00, 0xD7AF), // Hangul Syllables.
        (0xD7B0, 0xD7FF), // Hangul Jamo Extended-B.
        (0xF900, 0xFAFF), // CJK Compatibility Ideographs.
        (0xFB00, 0xFB4F), // Alphabetic Presentation Forms.
        (0xFB50, 0xFDFF), // Arabic Presentation Forms-A.
        (0xFE00, 0xFE0F), // Variation Selectors.
        (0xFE20, 0xFE2F), // Combining Half Marks.
        (0xFE30, 0xFE4F), // CJK Compatibility Forms.
        (0xFE50, 0xFE6F), // Small Form Variants.
        (0xFE70, 0xFEFF), // Arabic Presentation Forms-B.
        (0xFF00, 0xFFEF), // Halfwidth and Fullwidth Forms.
    ] {
        seed_char_range(seen, start, end);
    }

    for (start, end) in [
        (0x10000, 0x1007F), // Linear B Syllabary.
        (0x10080, 0x100FF), // Linear B Ideograms.
        (0x10100, 0x1013F), // Aegean Numbers.
        (0x10140, 0x1018F), // Ancient Greek Numbers.
        (0x10190, 0x101CF), // Ancient Symbols.
        (0x101D0, 0x101FF), // Phaistos Disc.
        (0x10280, 0x1029F), // Lycian.
        (0x102A0, 0x102DF), // Carian.
        (0x102E0, 0x102FF), // Coptic Epact Numbers.
        (0x10300, 0x1032F), // Old Italic.
        (0x10330, 0x1034F), // Gothic.
        (0x10350, 0x1037F), // Old Permic.
        (0x10380, 0x1039F), // Ugaritic.
        (0x103A0, 0x103DF), // Old Persian.
        (0x10400, 0x1044F), // Deseret.
        (0x10450, 0x1047F), // Shavian.
        (0x10480, 0x104AF), // Osmanya.
        (0x104B0, 0x104FF), // Osage.
        (0x10500, 0x1052F), // Elbasan.
        (0x10530, 0x1056F), // Caucasian Albanian.
        (0x10570, 0x105BF), // Vithkuqi.
        (0x10600, 0x1077F), // Linear A.
        (0x10780, 0x107BF), // Latin Extended-F.
        (0x10800, 0x1083F), // Cypriot Syllabary.
        (0x10840, 0x1085F), // Imperial Aramaic.
        (0x10860, 0x1087F), // Palmyrene.
        (0x10880, 0x108AF), // Nabataean.
        (0x108E0, 0x108FF), // Hatran.
        (0x10900, 0x1091F), // Phoenician.
        (0x10920, 0x1093F), // Lydian.
        (0x10980, 0x1099F), // Meroitic Hieroglyphs.
        (0x109A0, 0x109FF), // Meroitic Cursive.
        (0x10A00, 0x10A5F), // Kharoshthi.
        (0x10A60, 0x10A7F), // Old South Arabian.
        (0x10A80, 0x10A9F), // Old North Arabian.
        (0x10AC0, 0x10AFF), // Manichaean.
        (0x10B00, 0x10B3F), // Avestan.
        (0x10B40, 0x10B5F), // Inscriptional Parthian.
        (0x10B60, 0x10B7F), // Inscriptional Pahlavi.
        (0x10B80, 0x10BAF), // Psalter Pahlavi.
        (0x10C00, 0x10C4F), // Old Turkic.
        (0x10C80, 0x10CFF), // Old Hungarian.
        (0x10D00, 0x10D3F), // Hanifi Rohingya.
        (0x10E60, 0x10E7F), // Rumi Numeral Symbols.
        (0x10E80, 0x10EBF), // Yezidi.
        (0x10EC0, 0x10EFF), // Arabic Extended-C.
        (0x10F00, 0x10F2F), // Old Sogdian.
        (0x10F30, 0x10F6F), // Sogdian.
        (0x10F70, 0x10FAF), // Old Uyghur.
        (0x10FB0, 0x10FDF), // Chorasmian.
        (0x10FE0, 0x10FFF), // Elymaic.
        (0x11000, 0x1107F), // Brahmi.
        (0x11080, 0x110CF), // Kaithi.
        (0x110D0, 0x110FF), // Sora Sompeng.
        (0x11100, 0x1114F), // Chakma.
        (0x11150, 0x1117F), // Mahajani.
        (0x11180, 0x111DF), // Sharada.
        (0x111E0, 0x111FF), // Sinhala Archaic Numbers.
        (0x11200, 0x1124F), // Khojki.
        (0x11280, 0x112AF), // Multani.
        (0x112B0, 0x112FF), // Khudawadi.
        (0x11300, 0x1137F), // Grantha.
        (0x11400, 0x1147F), // Newa.
        (0x11480, 0x114DF), // Tirhuta.
        (0x11580, 0x115FF), // Siddham.
        (0x11600, 0x1165F), // Modi.
        (0x11660, 0x1167F), // Mongolian Supplement.
        (0x11680, 0x116CF), // Takri.
        (0x11700, 0x1174F), // Ahom.
        (0x11800, 0x1184F), // Dogra.
        (0x118A0, 0x118FF), // Warang Citi.
        (0x11900, 0x1195F), // Dives Akuru.
        (0x119A0, 0x119FF), // Nandinagari.
        (0x11A00, 0x11A4F), // Zanabazar Square.
        (0x11A50, 0x11AAF), // Soyombo.
        (0x11AB0, 0x11ABF), // Unified Canadian Aboriginal Syllabics Extended-A.
        (0x11AC0, 0x11AFF), // Pau Cin Hau.
        (0x11C00, 0x11C6F), // Bhaiksuki.
        (0x11C70, 0x11CBF), // Marchen.
        (0x11D00, 0x11D5F), // Masaram Gondi.
        (0x11D60, 0x11DAF), // Gunjala Gondi.
        (0x11EE0, 0x11EFF), // Makasar.
        (0x11F00, 0x11F5F), // Kawi.
        (0x11FB0, 0x11FBF), // Lisu Supplement.
        (0x11FC0, 0x11FFF), // Tamil Supplement.
        (0x12000, 0x123FF), // Cuneiform.
        (0x12400, 0x1247F), // Cuneiform Numbers and Punctuation.
        (0x12480, 0x1254F), // Early Dynastic Cuneiform.
        (0x16800, 0x16A3F), // Bamum Supplement.
        (0x16A40, 0x16A6F), // Mro.
        (0x16A70, 0x16ACF), // Tangsa.
        (0x16AD0, 0x16AFF), // Bassa Vah.
        (0x16B00, 0x16B8F), // Pahawh Hmong.
        (0x16E40, 0x16E9F), // Medefaidrin.
        (0x16F00, 0x16F9F), // Miao.
        (0x16FE0, 0x16FFF), // Ideographic Symbols and Punctuation.
        (0x17000, 0x187FF), // Tangut.
        (0x18800, 0x18AFF), // Tangut Components.
        (0x18B00, 0x18CFF), // Khitan Small Script.
        (0x18D00, 0x18D7F), // Tangut Supplement.
        (0x1AFF0, 0x1AFFF), // Kana Extended-B.
        (0x1B000, 0x1B0FF), // Kana Supplement.
        (0x1B100, 0x1B12F), // Kana Extended-A.
        (0x1B130, 0x1B16F), // Small Kana Extension.
        (0x1B170, 0x1B2FF), // Nushu.
        (0x1BC00, 0x1BC9F), // Duployan.
        (0x1BCA0, 0x1BCAF), // Shorthand Format Controls.
        (0x1CF00, 0x1CFCF), // Znamenny Musical Notation.
        (0x1D000, 0x1D0FF), // Byzantine Musical Symbols.
        (0x1D100, 0x1D1FF), // Musical Symbols.
        (0x1D200, 0x1D24F), // Ancient Greek Musical Notation.
        (0x1D2C0, 0x1D2DF), // Kaktovik Numerals.
        (0x1D2E0, 0x1D2FF), // Mayan Numerals.
        (0x1D300, 0x1D35F), // Tai Xuan Jing Symbols.
        (0x1D360, 0x1D37F), // Counting Rod Numerals.
        (0x1D400, 0x1D7FF), // Mathematical Alphanumeric Symbols.
        (0x1E000, 0x1E02F), // Glagolitic Supplement.
        (0x1E030, 0x1E08F), // Cyrillic Extended-D.
        (0x1E100, 0x1E14F), // Nyiakeng Puachue Hmong.
        (0x1E290, 0x1E2BF), // Toto.
        (0x1E2C0, 0x1E2FF), // Wancho.
        (0x1E4D0, 0x1E4FF), // Nag Mundari.
        (0x1E7E0, 0x1E7FF), // Ethiopic Extended-B.
        (0x1E800, 0x1E8DF), // Mende Kikakui.
        (0x1E900, 0x1E95F), // Adlam.
        (0x1EC70, 0x1ECBF), // Indic Siyaq Numbers.
        (0x1ED00, 0x1ED4F), // Ottoman Siyaq Numbers.
        (0x1EE00, 0x1EEFF), // Arabic Mathematical Alphabetic Symbols.
        (0x1F100, 0x1F1FF), // Enclosed Alphanumeric Supplement.
        (0x1F200, 0x1F2FF), // Enclosed Ideographic Supplement.
        (0x1F300, 0x1F5FF), // Miscellaneous Symbols and Pictographs.
        (0x1F600, 0x1F64F), // Emoticons.
        (0x1F650, 0x1F67F), // Ornamental Dingbats.
        (0x1F680, 0x1F6FF), // Transport and Map Symbols.
        (0x1F700, 0x1F77F), // Alchemical Symbols.
        (0x1F780, 0x1F7FF), // Geometric Shapes Extended.
        (0x1F800, 0x1F8FF), // Supplemental Arrows-C.
        (0x1F900, 0x1F9FF), // Supplemental Symbols and Pictographs.
        (0x1FA00, 0x1FA6F), // Chess Symbols.
        (0x1FA70, 0x1FAFF), // Symbols and Pictographs Extended-A.
        (0x1FB00, 0x1FBFF), // Symbols for Legacy Computing.
        (0x20000, 0x2A6DF), // CJK Unified Ideographs Extension B.
        (0x2A700, 0x2B73F), // CJK Unified Ideographs Extension C.
        (0x2B740, 0x2B81F), // CJK Unified Ideographs Extension D.
        (0x2B820, 0x2CEAF), // CJK Unified Ideographs Extension E.
        (0x2CEB0, 0x2EBEF), // CJK Unified Ideographs Extension F.
        (0x2F800, 0x2FA1F), // CJK Compatibility Ideographs Supplement.
        (0x30000, 0x3134F), // CJK Unified Ideographs Extension G.
        (0x31350, 0x323AF), // CJK Unified Ideographs Extension H.
    ] {
        seed_char_range(seen, start, end);
    }
}

fn seed_char_range(seen: &mut std::collections::BTreeSet<String>, start: u32, end: u32) {
    for codepoint in start..=end {
        if let Some(c) = char::from_u32(codepoint) {
            if !c.is_control() {
                seen.insert(c.to_string());
            }
        }
    }
}

// ── Errors ─────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum VocabError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_roundtrip() {
        let words = vec!["hello".to_string(), "world".to_string()];
        let phonemes = vec!["həˈloʊ".to_string()];
        let phones = vec!["hə.ˈloʊ".to_string()];
        let v = Vocab::build(&words, &phonemes, &phones);

        assert_eq!(v.get_id("<PAD>"), PAD_ID);
        assert_eq!(v.get_id("<UNK>"), UNK_ID);
        assert_eq!(v.get_id("<BOS>"), BOS_ID);

        let encoded = v.encode_string("hello");
        assert_eq!(encoded.len(), 5);
        let decoded = v.decode_ids(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn vocab_encodes_angle_bracket_controls_as_atomic_tokens() {
        let words =
            vec!["<task:orthography_to_phonology> <lang:eng> <repr:phonemes> disease".to_string()];
        let phonemes = vec!["dəˈziːz".to_string()];
        let v = Vocab::build(&words, &phonemes, &[]);

        let task_id = v.get_id("<task:orthography_to_phonology>");
        let lang_id = v.get_id("<lang:eng>");
        let repr_id = v.get_id("<repr:phonemes>");
        let align_id = v.get_id("<task:align>");
        assert_ne!(task_id, UNK_ID);
        assert_ne!(lang_id, UNK_ID);
        assert_ne!(repr_id, UNK_ID);
        assert_ne!(align_id, UNK_ID);

        let encoded =
            v.encode_string("<task:orthography_to_phonology> <lang:eng> <repr:phonemes> disease");
        assert_eq!(encoded[0], task_id);
        assert_eq!(encoded[2], lang_id);
        assert_eq!(encoded[4], repr_id);
    }

    #[test]
    fn vocab_seeds_broad_linguistic_ranges() {
        let v = Vocab::build(&[], &[], &[]);
        for token in [
            "<lang:lat>",
            "<lang:ell>",
            "<lang:grc>",
            "<lang:san>",
            "<repr:phonemes>",
            "<repr:phones>",
            "<repr:diaphonemes>",
            "<task:phonetic_realization>",
        ] {
            assert_ne!(v.get_id(token), UNK_ID, "{token} should be seeded");
        }
        for c in ['θ', 'ɲ', '͡', 'ᵻ', '᷄', 'ᾱ', 'क', 'ā'] {
            assert_ne!(v.get_id(&c.to_string()), UNK_ID, "{c} should be seeded");
        }
    }
}
