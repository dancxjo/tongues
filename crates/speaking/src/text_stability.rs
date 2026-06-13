//! Word-boundary-aware stability helpers for evolving transcript hypotheses.

pub fn shared_prefix_len(previous: &str, next: &str) -> usize {
    let mut len = 0;
    let mut previous_chars = previous.char_indices();
    let mut next_chars = next.char_indices();
    loop {
        match (previous_chars.next(), next_chars.next()) {
            (Some((idx, previous_char)), Some((_, next_char))) if previous_char == next_char => {
                len = idx + previous_char.len_utf8();
            }
            _ => break,
        }
    }
    len
}

pub fn stable_prefix_len(previous: &str, next: &str) -> usize {
    let shared = shared_prefix_len(previous, next);
    if shared == 0 {
        return 0;
    }
    if shared == previous.len() || shared == next.len() {
        return shared;
    }

    last_word_boundary_at_or_before(previous, shared)
        .zip(last_word_boundary_at_or_before(next, shared))
        .map(|(previous_boundary, next_boundary)| previous_boundary.min(next_boundary))
        .unwrap_or(shared)
}

fn last_word_boundary_at_or_before(text: &str, limit: usize) -> Option<usize> {
    let mut capped = limit.min(text.len());
    while capped > 0 && !text.is_char_boundary(capped) {
        capped -= 1;
    }
    if capped == 0 {
        return None;
    }

    let mut last_boundary = None;
    for (idx, ch) in text[..capped].char_indices() {
        if ch.is_whitespace() {
            last_boundary = Some(idx + ch.len_utf8());
        }
    }
    if capped < text.len() {
        let previous = text[..capped].chars().next_back();
        let next = text[capped..].chars().next();
        if let (Some(previous), Some(next)) = (previous, next)
            && (previous.is_whitespace() || next.is_whitespace())
        {
            last_boundary = Some(capped);
        }
    } else {
        last_boundary = Some(capped);
    }
    last_boundary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_prefix_detects_extension() {
        assert_eq!(stable_prefix_len("hello", "hello world"), "hello".len());
    }

    #[test]
    fn stable_prefix_retreats_to_word_boundary() {
        assert_eq!(
            stable_prefix_len("play music now", "play movie now"),
            "play ".len()
        );
    }

    #[test]
    fn stable_prefix_detects_restarted_head() {
        assert_eq!(stable_prefix_len("goodbye", "hello"), 0);
    }
}
