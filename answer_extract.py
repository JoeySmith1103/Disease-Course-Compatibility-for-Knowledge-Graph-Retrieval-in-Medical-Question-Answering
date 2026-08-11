#!/usr/bin/env python3
"""Pull the chosen option letter out of a reader's raw output.

Scoring an unextracted answer as wrong conflates two different failures — "the model was wrong"
and "the model answered in a shape the regex did not expect" — and the second is not uniformly
distributed across methods. In an audit of 76 unparsed answers, 27 carried a perfectly good letter
in a malformed wrapper and 23 of those were CORRECT, and they were concentrated in hykge/tog/
vanilla while walker and the raw dumps had none. Left unfixed, that is a systematic handicap
applied to some methods and not others.

The remaining 49 are genuine: 28 outputs stopped before answering, and 21 named something that is
not an option at all ("None of the above", a diagnosis absent from the list). Those stay wrong.

Answer text is deliberately matched only on an EXACT, unique option string. Substring matching was
measured to recover nothing on this data while being able to mis-assign ("CT of the chest without
contrast" is a substring of nothing else here, but that is luck, not a property) — the cost of a
wrong recovery is a silently corrupted accuracy, so the bar stays high.
"""
import re

# Ordered most-specific first. `[Answer]` leads because a template that asks for a
# [Reasoning]/[Answer] block will also contain the word "answer" inside the reasoning sentence,
# and matching the bracketed section first stops "the answer is unclear" being read as a verdict.
_PATTERNS = [
    r"\[Answer\]\s*:?\s*\**([A-J])\b",
    r"<a>\s*([A-J])\s*</a>",
    r"\\boxed\{\s*([A-J])\s*\}",
    # malformed wrappers: the model closed with a different tag letter than it opened, or bolded
    # the letter with <b>. The letter between the tags is still the answer.
    r"<\s*[a-z]\s*>\s*([A-J])\s*<\s*/\s*[a-z]?\s*>",
    # the letter became the tag name: "**Final answer: <e>**"
    r"final answer[^A-Za-z\n]{0,12}<\s*([A-Ja-j])\s*>",
    # truncated tag: "**Final answer: <G</a>"
    r"final answer[^A-Za-z\n]{0,12}<?\s*([A-J])\s*<",
    r"\*\*?Answer:?\*?\*?\s*:?\s*\*?\*?([A-J])\b",
    r"final answer[:\s]+(?:is\s+)?(?:\\boxed\{)?\*?\*?([A-J])\b",
    r"answer is[:\s]+(?:\\boxed\{)?\*?\*?([A-J])\b",
    r"\bthe answer\s*(?:is|:)\s*\*?\*?([A-J])\b",
    # a bare letter alone on the final line
    r"(?:^|\n)\s*\**\s*([A-J])\s*\**\s*$",
]
_COMPILED = [re.compile(p, re.I) for p in _PATTERNS]
_INTAG = re.compile(r"<a>\s*(.+?)\s*</a>", re.I | re.S)


def extract_letter(raw, options=None):
    """Letter, or None if the output never committed to one of `options`.

    options: {letter: text}. When given, a match is only accepted if the letter is actually on
    offer — otherwise a stray "B" in prose can be read as an answer to a 4-option question — and
    an exact option string is accepted as a last resort.
    """
    raw = raw or ""
    valid = set(options) if options else None
    for rx in _COMPILED:
        m = rx.search(raw)
        if m:
            lab = m.group(1).upper()
            if valid is None or lab in valid:
                return lab
    if options:
        m = _INTAG.search(raw)
        if m:
            txt = m.group(1).strip().lower()
            hit = [L for L, v in options.items() if (v or "").strip().lower() == txt]
            if len(hit) == 1:
                return hit[0]
    return None
