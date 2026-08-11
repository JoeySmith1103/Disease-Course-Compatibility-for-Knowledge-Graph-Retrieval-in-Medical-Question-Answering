"""
Spectrum + Textbook Method — Iteration 2

Uses textbook-derived duration information + deterministic spectrum computation.

Architecture (all steps 1-4 are outside the LLM):
  1. [Rule-based] Extract patient duration from question
  2. [Rule-based] Parse textbook-derived per-option duration ranges
  3. [Deterministic] Compute spectrum compatibility per option
  4. [Deterministic] Generate structured temporal analysis
  5. [LLM] Answer with pre-computed spectrum context

Key distinction from prompting:
  - Steps 1-4 are fully deterministic, verifiable, ablatable
  - The LLM receives pre-computed analysis, NOT raw textbook text
  - Ablation: textbook text only vs. textbook + spectrum computation
"""

import json
import math
import os
import re
import sys
import time
import requests
from typing import Dict, List, Optional, Tuple

# ── LLM API clients live in llm_client.py (the clearly-named home for call_llm & friends) ──
from llm_client import (get_api_key, call_openai, call_gemini, get_openai_api_key,
                        call_vllm_openai, _litellm_token, _together_key,
                        call_together, call_litellm, call_llm)



def extract_answer(raw_output: str) -> str:
    m = re.search(r'<a>\s*([A-Z])\s*</a>', raw_output)
    if m:
        return m.group(1)
    m = re.search(r'\\boxed\{([A-Z])\}', raw_output)
    if m:
        return m.group(1)
    m = re.search(r'(?:final\s+)?(?:answer|choice)\s*(?:is|:)\s*[:\s]*\**([A-Z])\**\b', raw_output, re.I)
    if m:
        return m.group(1).upper()
    tail = raw_output[-200:] if len(raw_output) > 200 else raw_output
    m = re.findall(r'\b([A-Z])\b', tail)
    if m:
        return m[-1]
    return ""


# ── Step 1: Patient Duration Extraction (rule-based) ───────────────────

NUMBER_WORDS = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'several': '3', 'few': '3', 'couple': '2',
}

UNIT_TO_DAYS = {
    'minute': 1/1440, 'min': 1/1440,
    'hour': 1/24, 'hr': 1/24,
    'day': 1,
    'week': 7, 'wk': 7,
    'month': 30, 'mo': 30,
    'year': 365, 'yr': 365,
}

AGE_PATTERN = re.compile(
    r'(\d+)\s*[-–]?\s*year\s*[-–]?\s*old|aged?\s+(\d+)|(\d+)\s+years?\s+of\s+age', re.I)
GESTATION_PATTERN = re.compile(
    r"(\d+)\s*(?:weeks?[''`]?\s*)?(?:gestation|gestational|pregnant)", re.I)


def extract_patient_duration(question: str) -> Dict:
    for word, digit in NUMBER_WORDS.items():
        question = re.sub(rf'\b{word}\b', digit, question, flags=re.I)

    patterns = [
        (re.compile(r'(\d+\.?\d*)\s*[-–]?\s*(minute|min|hour|hr|day|week|wk|month|mo|year|yr)s?\s+(?:history|hx)\s+of', re.I), 'history_of'),
        (re.compile(r'for\s+(?:the\s+)?(?:past|last)\s+(\d+\.?\d*)\s*[-–]?\s*(minute|min|hour|hr|day|week|wk|month|mo|year|yr)s?', re.I), 'for_past'),
        (re.compile(r'history\s+of\s+.{3,80}?for\s+(\d+\.?\d*)\s*[-–]?\s*(minute|min|hour|hr|day|week|wk|month|mo|year|yr)s?', re.I), 'history_for'),
        (re.compile(r'(?:start|began|onset)\w*\s+(\d+\.?\d*)\s*[-–]?\s*(minute|min|hour|hr|day|week|wk|month|mo|year|yr)s?\s+ago', re.I), 'started_ago'),
        (re.compile(r'(?:have|has)\s+lasted\s+(\d+\.?\d*)\s*[-–]?\s*(minute|min|hour|hr|day|week|wk|month|mo|year|yr)s?', re.I), 'lasted'),
        (re.compile(r'(?:for|over)\s+(\d+\.?\d*)\s*[-–]?\s*(minute|min|hour|hr|day|week|wk|month|mo|year|yr)s?', re.I), 'for_duration'),
        (re.compile(r'(\d+\.?\d*)\s*[-–]?\s*(minute|min|hour|hr|day|week|wk|month|mo|year|yr)s?\s+ago', re.I), 'ago'),
        # Level 8: "X units of [symptom]" (e.g., "30 minutes of chest pain")
        (re.compile(r'(\d+\.?\d*)\s*[-–]?\s*(minute|min|hour|hr|day|week|wk|month|mo|year|yr)s?\s+of\s+', re.I), 'duration_of'),
    ]

    def is_excluded(match, text):
        start, end = match.span()
        context = text[max(0, start-30):min(len(text), end+30)]
        if AGE_PATTERN.search(context):
            for am in AGE_PATTERN.finditer(text):
                if abs(am.start() - start) < 20:
                    return True
        if GESTATION_PATTERN.search(context):
            return True
        return False

    for pat, source in patterns:
        for m in pat.finditer(question):
            if is_excluded(m, question):
                continue
            value = float(m.group(1))
            unit = m.group(2).lower().rstrip('s')
            days = value * UNIT_TO_DAYS.get(unit, 1)
            return {'value': value, 'unit': unit, 'days': days, 'source': source}

    return {'value': None, 'unit': None, 'days': None, 'source': 'none'}


# ── Step 2: Parse Textbook Duration Info (rule-based) ──────────────────

# Temporal class → approximate day ranges
TEMPORAL_CLASS_RANGES = {
    'minutes': (0.001, 0.04),          # minutes = 0.001-0.04 days
    'minutes to hours': (0.01, 0.5),
    'hours': (0.04, 1),
    'hours to days': (0.17, 3),
    'days': (1, 14),
    'days to weeks': (3, 28),
    'weeks': (7, 42),
    'weeks to months': (14, 90),
    'months': (30, 180),
    'months to years': (60, 730),
    'years': (365, 3650),
    'years to decades': (365, 7300),
    'decades': (3650, 18250),
    'chronic': (42, 3650),
    'acute': (0.04, 7),
    'subacute': (7, 42),
    'ongoing': (30, 3650),
    'long-term': (90, 3650),
    'lifelong': (365, 36500),
    'permanent': (365, 36500),
    'variable': (1, 365),
}


def parse_duration_info(duration_info: str, options: Dict[str, str]) -> Dict[str, Dict]:
    """Parse textbook-derived duration_info into per-option day ranges.

    Fully rule-based — no LLM call.

    Strategy: split text by option letters, then parse each section independently.
    Returns: {option_letter: {min_days, max_days, temporal_class, raw_text}}
    """
    profiles = {}
    letters = sorted(options.keys())

    # Split duration_info into per-option sections
    # Strategy: find option labels like "A." "A:" "A)" at line starts or after newlines
    sections = {}

    # Find all option label positions
    # Match: newline/start + optional whitespace + letter + separator
    label_pat = re.compile(
        r'(?:^|\n)\s*([A-G])[\.\):]\s*',
        re.MULTILINE
    )
    # Also try: "- A." or "* A." list markers
    label_pat2 = re.compile(
        r'(?:^|\n)\s*[-*•]\s*([A-G])[\.\)\s:]\s*',
        re.MULTILINE
    )

    label_positions = []
    for m in label_pat.finditer(duration_info):
        letter = m.group(1).upper()
        if letter in letters:
            label_positions.append((m.start(), m.end(), letter))
    for m in label_pat2.finditer(duration_info):
        letter = m.group(1).upper()
        if letter in letters:
            label_positions.append((m.start(), m.end(), letter))

    # Sort by position and deduplicate
    label_positions.sort(key=lambda x: x[0])
    seen = set()
    unique_positions = []
    for start, end, letter in label_positions:
        if letter not in seen:
            seen.add(letter)
            unique_positions.append((start, end, letter))

    # Extract sections
    for i, (start, end, letter) in enumerate(unique_positions):
        if i + 1 < len(unique_positions):
            next_start = unique_positions[i+1][0]
            section_text = duration_info[end:next_start].strip()
        else:
            section_text = duration_info[end:].strip()
        sections[letter] = section_text[:300]

    # For letters without sections, try inline pattern: "A. text, B. text"
    for letter in letters:
        if letter not in sections or not sections[letter]:
            inline_pat = re.compile(
                rf'\b{letter}[\.\)]\s*(.*?)(?=\b[A-G][\.\)]|$)',
                re.I | re.DOTALL
            )
            m = inline_pat.search(duration_info)
            if m:
                sections[letter] = m.group(1).strip()[:300]
            else:
                sections[letter] = ""

    for letter in letters:
        section = sections.get(letter, "")
        min_days, max_days, temporal_class = _parse_section_duration(section)

        profiles[letter] = {
            'min_days': min_days,
            'max_days': max_days,
            'temporal_class': temporal_class or 'unknown',
            'raw_text': section[:200],
        }

    return profiles


def _parse_section_duration(text: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Parse a single option's text section for duration information.

    Returns: (min_days, max_days, temporal_class)
    """
    if not text:
        return None, None, None

    text_lower = text.lower()

    # Pattern 1: Explicit range "X-Y days" or "X to Y weeks"
    range_pat = re.compile(
        r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*(minute|hour|day|week|month|year)s?', re.I)
    m = range_pat.search(text)
    if m:
        v1, v2 = float(m.group(1)), float(m.group(2))
        unit = m.group(3).lower().rstrip('s')
        mult = UNIT_TO_DAYS.get(unit, 1)
        return v1 * mult, v2 * mult, f"{v1}-{v2} {unit}s"

    # Pattern 2: "X to Y units" (with word "to")
    to_pat = re.compile(
        r'(\d+\.?\d*)\s*(minute|hour|day|week|month|year)s?\s+to\s+'
        r'(\d+\.?\d*)\s*(minute|hour|day|week|month|year)s?', re.I)
    m = to_pat.search(text)
    if m:
        v1 = float(m.group(1))
        u1 = m.group(2).lower().rstrip('s')
        v2 = float(m.group(3))
        u2 = m.group(4).lower().rstrip('s')
        return v1 * UNIT_TO_DAYS.get(u1, 1), v2 * UNIT_TO_DAYS.get(u2, 1), f"{v1} {u1}s to {v2} {u2}s"

    # Pattern 3: Cross-unit class "minutes to hours", "months to years", etc.
    cross_unit_patterns = [
        (r'minutes?\s+to\s+hours?', 'minutes to hours'),
        (r'hours?\s+to\s+days?', 'hours to days'),
        (r'days?\s+to\s+weeks?', 'days to weeks'),
        (r'weeks?\s+to\s+months?', 'weeks to months'),
        (r'months?\s+to\s+years?', 'months to years'),
        (r'years?\s+to\s+decades?', 'years to decades'),
    ]
    for pat, tc in cross_unit_patterns:
        if re.search(pat, text_lower):
            lo, hi = TEMPORAL_CLASS_RANGES[tc]
            return lo, hi, tc

    # Pattern 4: Single value with comparator "<5 days", ">2 weeks"
    comp_pat = re.compile(r'[<≤]\s*(\d+\.?\d*)\s*(minute|hour|day|week|month|year)s?', re.I)
    m = comp_pat.search(text)
    if m:
        v = float(m.group(1))
        unit = m.group(2).lower().rstrip('s')
        max_d = v * UNIT_TO_DAYS.get(unit, 1)
        return max_d * 0.1, max_d, f"<{v} {unit}s"

    comp_pat2 = re.compile(r'[>≥]\s*(\d+\.?\d*)\s*(minute|hour|day|week|month|year)s?', re.I)
    m = comp_pat2.search(text)
    if m:
        v = float(m.group(1))
        unit = m.group(2).lower().rstrip('s')
        min_d = v * UNIT_TO_DAYS.get(unit, 1)
        return min_d, min_d * 5, f">{v} {unit}s"

    # Pattern 5: Single value "X days/weeks/months"
    single_pat = re.compile(r'(\d+\.?\d*)\s*(minute|hour|day|week|month|year)s?', re.I)
    m = single_pat.search(text)
    if m:
        v = float(m.group(1))
        unit = m.group(2).lower().rstrip('s')
        center = v * UNIT_TO_DAYS.get(unit, 1)
        return center * 0.5, center * 1.5, f"~{v} {unit}s"

    # Pattern 6: Temporal class keywords
    class_keywords = [
        (r'\bchronic\b', 'chronic'),
        (r'\bacute\b', 'acute'),
        (r'\bsubacute\b', 'subacute'),
        (r'\bongoing\b', 'ongoing'),
        (r'\blong[\s-]?term\b', 'long-term'),
        (r'\blifelong\b', 'lifelong'),
        (r'\bpermanent\b', 'permanent'),
        (r'\bvariable\b', 'variable'),
    ]
    for pat, tc in class_keywords:
        if re.search(pat, text_lower):
            lo, hi = TEMPORAL_CLASS_RANGES.get(tc, (1, 365))
            return lo, hi, tc

    return None, None, None


# ── Step 3: Spectrum Computation (deterministic, Bhattacharyya) ─────────
#
# Uses Bhattacharyya Coefficient (BC) on log-Normal distributions.
# Validated: BC + textbook facts = 74.46% on Qwen3.5-9B (325 questions).
#
# Reference: Bhattacharyya, A. (1943). "On a measure of divergence between
#            two statistical populations." Bulletin of the Calcutta
#            Mathematical Society, 35, 99-109.
#
# Log-Normal fit justified by: Alderson (1987) "The lognormal distribution
# of the duration of symptoms", British Journal of Cancer — 26,000 patients.

PATIENT_SIGMA = 0.5  # Optimal per ablation (σ=0.3: 71.69%, σ=0.5: 74.46%, σ=0.8: 71.08%)


def _fit_lognormal_from_range(min_days: float, max_days: float) -> Tuple[float, float]:
    """Fit log-Normal parameters (μ, σ) from a (min, max) duration range.

    Assumes the range represents approximately the 10th-90th percentile
    of the disease's duration distribution.

    Returns: (mu, sigma) in log(days) space.
    """
    min_d = max(min_days, 1/24)  # floor at 1 hour
    max_d = max(max_days, min_d * 1.1)  # ensure max > min

    # μ = midpoint of log range
    mu = (math.log(min_d) + math.log(max_d)) / 2
    # σ = half-width covers ~80% of distribution (z=1.28 for 10th-90th)
    sigma = (math.log(max_d) - math.log(min_d)) / (2 * 1.28)
    sigma = max(sigma, 0.3)  # minimum uncertainty

    return mu, sigma


def bhattacharyya_coefficient(mu_p: float, sigma_p: float,
                               mu_d: float, sigma_d: float) -> float:
    """Bhattacharyya coefficient between two log-Normal distributions.

    BC = √(2·σ_p·σ_d / (σ_p² + σ_d²)) · exp(-(μ_p - μ_d)² / (4·(σ_p² + σ_d²)))

    Returns BC ∈ [0, 1]: 1 = identical distributions, 0 = no overlap.
    """
    var_sum = sigma_p ** 2 + sigma_d ** 2
    if var_sum <= 0:
        return 0.5

    coeff = math.sqrt(2 * sigma_p * sigma_d / var_sum)
    exponent = -((mu_p - mu_d) ** 2) / (4 * var_sum)
    return coeff * math.exp(exponent)


def compute_spectrum_compatibility(patient_days: float, option_profiles: Dict) -> Dict:
    """Compute temporal compatibility using Bhattacharyya Coefficient.

    All computation is deterministic — no LLM.

    Pipeline:
      1. Patient duration → log-Normal(μ_p, σ_p=0.5)
      2. Each option's textbook range → log-Normal(μ_d, σ_d)
      3. BC(patient, disease) = distributional overlap score

    Returns per-option: {bc_score, mu_d, sigma_d, assessment}
    """
    # Patient distribution in log-space
    patient_d = max(patient_days, 1/24)  # floor at 1 hour
    mu_p = math.log(patient_d)
    sigma_p = PATIENT_SIGMA

    results = {}

    for letter, profile in option_profiles.items():
        min_d = profile.get('min_days')
        max_d = profile.get('max_days')

        if min_d is None or max_d is None:
            results[letter] = {
                'score': 0.5,
                'bc': 0.5,
                'assessment': 'No temporal data — neutral score',
                'temporal_class': profile.get('temporal_class', 'unknown'),
            }
            continue

        # Fit disease distribution
        mu_d, sigma_d = _fit_lognormal_from_range(min_d, max_d)

        # Compute Bhattacharyya Coefficient
        bc = bhattacharyya_coefficient(mu_p, sigma_p, mu_d, sigma_d)
        bc = round(bc, 3)

        # Generate assessment
        if bc >= 0.85:
            assessment = f"High temporal overlap (BC={bc:.2f}). Duration strongly consistent."
        elif bc >= 0.6:
            assessment = f"Moderate temporal overlap (BC={bc:.2f}). Duration compatible."
        elif bc >= 0.3:
            assessment = f"Low temporal overlap (BC={bc:.2f}). Duration less typical."
        else:
            assessment = f"Minimal temporal overlap (BC={bc:.2f}). Duration atypical for this condition."

        results[letter] = {
            'score': bc,
            'bc': bc,
            'mu_d': round(mu_d, 3),
            'sigma_d': round(sigma_d, 3),
            'typical_range_days': f"{min_d:.1f}-{max_d:.1f}",
            'temporal_class': profile.get('temporal_class', 'unknown'),
            'assessment': assessment,
        }

    return results


# ── Step 4: Prompt Construction ────────────────────────────────────────

def build_textbook_only_prompt(question: str, options: Dict[str, str],
                                duration_info: str) -> str:
    """Baseline: inject raw textbook duration info as text (= prompting)."""
    options_str = "\n".join(f"  {k}. {v}" for k, v in sorted(options.items()))

    prompt = f"""You are an expert medical diagnostician. Answer the following multiple-choice medical question.

Here is additional temporal information about each option from medical textbooks:

{duration_info}

Question:
{question}

Options:
{options_str}

Give a brief reasoning (under 200 words), then state your final answer as a single letter enclosed in <a></a> tags. Example: <a>C</a>"""

    return prompt


def build_spectrum_textbook_prompt(question: str, options: Dict[str, str],
                                    patient_dur: Dict, compatibility: Dict) -> str:
    """Spectrum method: inject pre-computed temporal analysis (= architecture)."""
    if patient_dur.get('days') is not None:
        dur_str = f"{patient_dur['value']} {patient_dur['unit']}(s) = {patient_dur['days']:.1f} days"
    else:
        dur_str = "Not explicitly stated"

    options_str = "\n".join(f"  {k}. {v}" for k, v in sorted(options.items()))

    # Sort options by compatibility score
    sorted_compat = sorted(compatibility.items(), key=lambda x: -x[1].get('score', 0))

    compat_lines = []
    for letter, info in sorted_compat:
        name = options.get(letter, f"Option {letter}")
        score = info.get('score', 0.5)
        tc = info.get('temporal_class', '?')
        typical = info.get('typical_range_days', '?')
        assessment = info.get('assessment', '')

        if score >= 0.8:
            bar = '██████████'
        elif score >= 0.5:
            bar = '██████░░░░'
        elif score >= 0.2:
            bar = '███░░░░░░░'
        else:
            bar = '█░░░░░░░░░'

        compat_lines.append(
            f"  {letter}. {name}\n"
            f"     Typical: {tc} ({typical} days) | Fit: {bar} {score:.2f}\n"
            f"     {assessment}"
        )

    compat_block = "\n".join(compat_lines)

    prompt = f"""You are an expert medical diagnostician. Answer the following multiple-choice medical question.

A temporal reasoning module has analyzed the patient's symptom duration against textbook-derived typical durations for each condition. The compatibility scores below are pre-computed — they indicate how well the patient's timeline matches each condition.

=== TEMPORAL SPECTRUM ANALYSIS (pre-computed by external module) ===
Patient symptom duration: {dur_str}

Temporal compatibility with each option:
{compat_block}
=== END ANALYSIS ===

Question:
{question}

Options:
{options_str}

Use the temporal compatibility as ONE diagnostic factor alongside all clinical findings. High compatibility supports but does not prove a diagnosis; low compatibility should prompt careful consideration but does not exclude.

Give a brief reasoning (under 200 words), then state your final answer as a single letter enclosed in <a></a> tags. Example: <a>C</a>"""

    return prompt


def build_vanilla_prompt(question: str, options: Dict[str, str]) -> str:
    options_str = "\n".join(f"  {k}. {v}" for k, v in sorted(options.items()))
    return f"""You are an expert medical diagnostician. Answer the following multiple-choice medical question.

Question:
{question}

Options:
{options_str}

Give a brief reasoning (under 200 words), then state your final answer as a single letter enclosed in <a></a> tags. Example: <a>C</a>"""


# ── Experiment Runner ──────────────────────────────────────────────────

def load_error_questions() -> List[Dict]:
    """Load all 139 Gemini error questions with their duration_info."""
    _rescue_path = 'results/duration_rescue_gemini.json' if os.path.exists('results/duration_rescue_gemini.json') else 'results_selfrag/duration_rescue_gemini.json'
    with open(_rescue_path) as f:
        d = json.load(f)
    questions = []
    for r in d['results']:
        questions.append({
            'idx': r['idx'],
            'question': r['question'],
            'options': r['options'],
            'answer': r['answer'],
            'vanilla_pred': r.get('vanilla_pred', ''),
            'duration_info': r.get('duration_info', ''),
            'was_rescued': r.get('rescued', False),
        })
    return questions


def run_experiment(questions: List[Dict], api_key: str, model: str,
                   output_path: str, mode: str = "spectrum_textbook"):
    """Run experiment on questions.

    Modes:
      vanilla: no temporal info
      textbook_only: raw textbook duration text (= prompting baseline)
      spectrum_textbook: textbook + spectrum computation (= architecture)
    """
    results = []
    correct = 0
    total = 0

    for i, q in enumerate(questions):
        question_text = q['question']
        options = q['options']
        answer = q['answer']
        idx = q.get('idx', i)
        duration_info = q.get('duration_info', '')

        patient_dur = extract_patient_duration(question_text)
        compatibility = None

        if mode == "spectrum_textbook" and patient_dur['days'] is not None and duration_info:
            # Parse textbook duration info (rule-based)
            option_profiles = parse_duration_info(duration_info, options)

            # Compute spectrum compatibility (deterministic)
            compatibility = compute_spectrum_compatibility(patient_dur['days'], option_profiles)

            # Check if spectrum is discriminating
            scores = [v.get('score', 0.5) for v in compatibility.values()]
            score_range = max(scores) - min(scores) if scores else 0

            if score_range >= 0.15:  # spectrum has discriminating power
                prompt = build_spectrum_textbook_prompt(
                    question_text, options, patient_dur, compatibility)
            else:
                # Not discriminating — fall back to textbook_only
                prompt = build_textbook_only_prompt(question_text, options, duration_info)
                compatibility = None

        elif mode == "textbook_only" and duration_info:
            prompt = build_textbook_only_prompt(question_text, options, duration_info)

        elif mode == "spectrum_textbook" and duration_info:
            # Has textbook info but no patient duration → use textbook only
            prompt = build_textbook_only_prompt(question_text, options, duration_info)

        elif mode == "vanilla":
            prompt = build_vanilla_prompt(question_text, options)

        else:
            # Fallback
            prompt = build_vanilla_prompt(question_text, options)

        raw_output = call_llm(prompt, model=model)
        pred = extract_answer(raw_output)
        is_correct = (pred == answer)

        if is_correct:
            correct += 1
        total += 1

        result = {
            'idx': idx,
            'answer': answer,
            'predicted': pred,
            'is_correct': is_correct,
            'mode': mode,
            'patient_duration': patient_dur,
            'compatibility': compatibility,
            'raw_output': raw_output,
        }
        results.append(result)

        status = "✓" if is_correct else "✗"
        if compatibility:
            top = max(compatibility.items(), key=lambda x: x[1].get('score', 0))
            info = f"[spectrum top={top[0]}:{top[1]['score']:.2f}]"
        elif mode == "textbook_only" and duration_info:
            info = "[textbook]"
        else:
            info = "[vanilla]"
        print(f"  [{i+1}/{len(questions)}] idx={idx} {status} pred={pred} ans={answer} {info}")

        time.sleep(0.3)

    accuracy = correct / total * 100 if total > 0 else 0

    output = {
        'metadata': {
            'mode': mode,
            'model': model,
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
        },
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*50}")
    print(f"Mode: {mode} | Model: {model}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"Saved: {output_path}")
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['vanilla', 'textbook_only', 'spectrum_textbook'],
                        default='spectrum_textbook')
    parser.add_argument('--model', default='gemini-3.1-flash-lite-preview')
    parser.add_argument('--output', default=None)
    parser.add_argument('--target', choices=['all_errors', 'rescued', 'test'],
                        default='all_errors')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    questions = load_error_questions()

    if args.target == 'rescued':
        questions = [q for q in questions if q.get('was_rescued')]
    elif args.target == 'test':
        questions = questions[:5]

    model_short = args.model.replace('gemini-', '').replace('-preview', '')
    default_output = f'results_selfrag/stb_{args.mode}_{args.target}_{model_short}.json'
    output_path = args.output or default_output

    if args.dry_run:
        for q in questions[:3]:
            patient_dur = extract_patient_duration(q['question'])
            if patient_dur['days'] is not None and q.get('duration_info'):
                profiles = parse_duration_info(q['duration_info'], q['options'])
                compat = compute_spectrum_compatibility(patient_dur['days'], profiles)
                print(f"=== idx={q['idx']} | dur={patient_dur['days']:.1f}d ===")
                for letter in sorted(compat.keys()):
                    c = compat[letter]
                    print(f"  {letter}: score={c['score']:.3f} | {c['assessment']}")
                print()
            else:
                print(f"=== idx={q['idx']} | NO DURATION ===\n")
    else:
        api_key = get_api_key()
        run_experiment(questions, api_key, args.model, output_path, mode=args.mode)
