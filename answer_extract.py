#!/usr/bin/env python3
"""Pull the chosen option letter out of a reader's raw output.

Unparseable answers are scored wrong, but they are also reported separately. The extractor is
therefore intentionally conservative: it recovers clear formatting mistakes, while avoiding broad
substring matching that could silently assign the wrong option.

`options` may be passed as `{letter: text}`. When present, a recovered letter is accepted only if
that letter exists for the question. This matters for mixed workloads: 1273 is A-E, MMLU/MedMCQA
are A-D, and 329 can contain A-L.
"""
import re

DEFAULT_ANSWER_LETTERS = "ABCDEFGHIJKL"


def _normalise_options(options):
    if not options:
        return None
    return {str(k).upper(): v for k, v in options.items() if str(k).strip()}


def _valid_letters(options=None):
    opts = _normalise_options(options)
    return set(opts) if opts else set(DEFAULT_ANSWER_LETTERS)


def _letter_class(valid):
    return "".join(re.escape(x) for x in sorted(valid))


def _unique_letter(candidates, valid):
    letters = [c.upper() for c in candidates if c and c.upper() in valid]
    return letters[0] if letters and len(set(letters)) == 1 else None


def _value_near_start(text, valid):
    text = (text or "").replace(chr(13), chr(10)).strip()
    if not text:
        return None

    noise = "*_`()[]{}<>:;,.#-" + chr(34) + chr(39)
    for ch in noise:
        text = text.replace(ch, " ")
    parts = text.split()

    while parts and parts[0].lower() in ("is", "would", "should", "be"):
        parts = parts[1:]
    if parts and parts[0].lower() in ("option", "choice", "letter"):
        parts = parts[1:]
    while parts and parts[0].lower() in ("is", "would", "should", "be"):
        parts = parts[1:]

    if not parts:
        return None
    token = parts[0].upper()
    if len(token) != 1 or token not in valid:
        return None
    if len(parts) == 1 or parts[1].lower() in (
            "because", "as", "is", "would", "should", "seems", "fits", "fit", "best"):
        return token
    return None


def _values_from_tag(raw, tag, valid):
    low = raw.lower()
    open_tag = "<" + tag + ">"
    close_tag = "</" + tag + ">"
    values = []
    start = 0
    while True:
        i = low.find(open_tag, start)
        if i < 0:
            break
        j = low.find(close_tag, i + len(open_tag))
        if j < 0:
            break
        val = _value_near_start(raw[i + len(open_tag):j], valid)
        if val:
            values.append(val)
        start = j + len(close_tag)
    return values


def _last_answer_marker(raw):
    low = raw.lower()
    markers = ["[answer]", "answer]", "answer:", "answer -", "answer-"]
    best_pos = -1
    best_len = 0
    for marker in markers:
        start = 0
        while True:
            i = low.find(marker, start)
            if i < 0:
                break
            if i >= best_pos:
                best_pos = i
                best_len = len(marker)
            start = i + 1
    return best_pos, best_len


def _value_after_option_word(text, valid):
    low = text.lower()
    for key in ("option", "choice", "letter"):
        j = low.find(key)
        if j >= 0:
            val = _value_near_start(text[j + len(key):j + len(key) + 50], valid)
            if val:
                return val
    return None


def _values_after_phrases(text, phrases, valid, window=220):
    low = text.lower()
    values = []
    for phrase in phrases:
        start = 0
        while True:
            i = low.find(phrase, start)
            if i < 0:
                break
            chunk = text[i + len(phrase):i + len(phrase) + window]
            val = _value_near_start(chunk, valid) or _value_after_option_word(chunk, valid)
            if val:
                values.append(val)
            start = i + len(phrase)
    return values


def _before_next_section(text):
    text = text or ""
    m = re.search(r"\n\s*\[[A-Za-z][A-Za-z _-]{0,40}\]", text)
    return text[:m.start()] if m else text


def _extract_from_final_context(text, valid):
    tail = (text or "")[-1200:]
    phrases = ("final answer", "the answer is", "correct answer", "best answer",
               "best choice", "best fit", "therefore", "thus", "hence",
               "overall", "in conclusion", "i choose", "i would choose",
               "i pick", "i select", "i lean towards")
    value = _unique_letter(_values_after_phrases(tail, phrases, valid), valid)
    if value:
        return value

    cls = _letter_class(valid)
    matches = re.findall(
        rf"(?:^|[\s.;:,])(?:(?:so|therefore|thus|hence|final(?:ly)?|overall)\s+)?"
        rf"(?:the\s+)?(?:correct\s+|best\s+)?(?:answer|choice)\s*"
        rf"(?:is|would\s+be|should\s+be|=|:)?\s*[\[(<]*\s*([{cls}])\b",
        tail,
        flags=re.IGNORECASE,
    )
    return matches[-1].upper() if matches else None


def _regex_patterns(valid):
    cls = _letter_class(valid)
    return [re.compile(p, re.I) for p in [
        rf"\[Answer\]\s*:?\s*\**([{cls}])\b",
        rf"<a>\s*([{cls}])\s*</a>",
        rf"\\boxed\{{\s*([{cls}])\s*\}}",
        rf"<\s*[a-z]\s*>\s*([{cls}])\s*<\s*/\s*[a-z]?\s*>",
        rf"final answer[^A-Za-z\n]{{0,12}}<\s*([{cls.lower()}{cls}])\s*>",
        rf"final answer[^A-Za-z\n]{{0,12}}<?\s*([{cls}])\s*<",
        rf"\*\*?Answer:?\*?\*?\s*:?\s*\*?\*?([{cls}])\b",
        rf"final answer[:\s]+(?:is\s+)?(?:\\boxed\{{)?\*?\*?([{cls}])\b",
        rf"answer is[:\s]+(?:\\boxed\{{)?\*?\*?([{cls}])\b",
        rf"\bthe answer\s*(?:is|:)\s*\*?\*?([{cls}])\b",
        rf"(?:^|\n)\s*\**\s*([{cls}])\s*\**\s*$",
    ]]


def extract_letter(raw, options=None):
    """Return the chosen option letter, or None if the output never clearly committed."""
    raw = raw or ""
    valid = _valid_letters(options)
    if not raw.strip():
        return None

    direct = _value_near_start(raw[:80], valid)
    if direct:
        return direct

    tagged = _unique_letter(_values_from_tag(raw, "a", valid) + _values_from_tag(raw, "answer", valid), valid)
    if tagged:
        return tagged

    for rx in _regex_patterns(valid):
        m = rx.search(raw)
        if m:
            lab = m.group(1).upper()
            if lab in valid:
                return lab

    pos, size = _last_answer_marker(raw)
    if pos >= 0:
        direct = _value_near_start(_before_next_section(raw[pos + size:pos + size + 240]), valid)
        if direct:
            return direct
        context = _extract_from_final_context(raw[:pos], valid)
        if context:
            return context

    labeled_phrases = ("answer:", "final answer", "the answer is", "correct answer",
                       "the correct answer is", "best answer", "best choice")
    labeled = _unique_letter(_values_after_phrases(raw, labeled_phrases, valid), valid)
    if labeled:
        return labeled

    final_pos = raw.lower().rfind("assistantfinal")
    if final_pos >= 0:
        final_text = raw[final_pos + len("assistantfinal"):]
        direct = _value_near_start(final_text[:120], valid)
        if direct:
            return direct
        final_context = _extract_from_final_context(final_text, valid)
        if final_context:
            return final_context

    tail_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if tail_lines:
        tail_direct = _value_near_start(tail_lines[-1], valid)
        if tail_direct:
            return tail_direct

    opts = _normalise_options(options)
    if opts:
        m = re.search(r"<a>\s*(.+?)\s*</a>", raw, re.I | re.S)
        if m:
            txt = m.group(1).strip().lower()
            hit = [L for L, v in opts.items() if (v or "").strip().lower() == txt]
            if len(hit) == 1:
                return hit[0]

    return _extract_from_final_context(raw, valid)
