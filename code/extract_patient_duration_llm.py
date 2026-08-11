"""
LLM-based role-tagged patient duration extraction.

Replaces the brittle regex `extract_patient_duration` for production use.
The LLM is asked to enumerate *all* temporal expressions in a clinical vignette
and classify each by its role. Only `chief_complaint` duration feeds ϕ matching.

Role taxonomy (fixed):
    chief_complaint      — duration of the primary presenting symptom / event
    onset_to_presentation — same, when framed as onset
    past_medical_history  — prior episodes / diagnoses / hospitalizations
    exposure              — travel / sexual / occupational / medication exposure
    age                   — patient age or age-related
    procedure             — diagnostic or therapeutic procedure duration
    medication            — ongoing drug therapy duration
    observation_window    — "since admission", "over the past 24 hours in ICU"
    other                 — gestation, follow-up intervals, etc.

Only `chief_complaint` and `onset_to_presentation` are considered temporally-
relevant for downstream ϕ matching against disease symptomatic-course priors.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llm_client import call_llm   # LLM routing lives in llm_client (pipeline-internal)


# ── Role taxonomy ────────────────────────────────────────────────────────

PHI_RELEVANT_ROLES = {'chief_complaint', 'onset_to_presentation'}

ROLE_SET = [
    'chief_complaint', 'onset_to_presentation',
    'past_medical_history', 'exposure', 'age',
    'procedure', 'medication', 'observation_window', 'other',
]


_UNIT_DAYS = {
    'minute': 1/1440, 'min': 1/1440, 'm': 1/1440,
    'hour': 1/24, 'hr': 1/24, 'h': 1/24,
    'day': 1.0, 'd': 1.0,
    'week': 7.0, 'wk': 7.0, 'w': 7.0,
    'month': 30.0, 'mo': 30.0,
    'year': 365.0, 'yr': 365.0, 'y': 365.0,
}

_NUMBER_WORDS = {
    'a': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
    'several': 3, 'few': 3, 'couple': 2,
}


# ── Prompt ───────────────────────────────────────────────────────────────

def _build_extraction_prompt(question: str) -> str:
    return f"""You are a clinical information extractor. Given a medical vignette, list EVERY temporal expression (duration, interval, age, timing) in the text. For each, classify by clinical role.

Role taxonomy:
  chief_complaint        — duration of the primary presenting symptom / reason for visit
  onset_to_presentation  — primary symptom framed as "started / began X ago"
  past_medical_history   — prior episodes, hospitalizations, old diagnoses
  exposure               — travel, sexual, occupational, dietary, or medication exposure
  age                    — patient or family member age
  procedure              — diagnostic or therapeutic procedure duration
  medication             — ongoing drug therapy duration
  observation_window     — "since admission", monitoring interval
  other                  — gestation, follow-up plan, study period, etc.

Respond with a JSON ARRAY. Each element has:
  span       — the exact verbatim phrase
  role       — one role from above
  value      — numeric magnitude (null if qualitative)
  unit       — "hour" | "day" | "week" | "month" | "year" | null
  days       — numeric duration in days (null if qualitative / age)

Vignette:
\"\"\"{question}\"\"\"

Output ONLY the JSON array. No prose, no code fence.
"""


# ── Parsing helpers ──────────────────────────────────────────────────────

def _phrase_to_days(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None or unit is None:
        return None
    unit = unit.lower().rstrip('s')
    if unit not in _UNIT_DAYS:
        return None
    try:
        return float(value) * _UNIT_DAYS[unit]
    except (TypeError, ValueError):
        return None


def _normalize_entry(entry: dict) -> dict:
    role = entry.get('role', 'other')
    if role not in ROLE_SET:
        role = 'other'
    value = entry.get('value')
    unit = entry.get('unit')
    days = entry.get('days')
    if days is None:
        days = _phrase_to_days(value, unit)
    return {
        'span': entry.get('span', ''),
        'role': role,
        'value': value,
        'unit': unit,
        'days': days,
    }


def _parse_json_array(raw: str) -> List[dict]:
    """Best-effort extraction of a JSON array from an LLM response."""
    if not raw:
        return []
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'```\s*$', '', cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\[[\s\S]*\]', cleaned)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return [_normalize_entry(e) for e in data if isinstance(e, dict)]


# ── Public API ───────────────────────────────────────────────────────────

def extract_patient_temporal_spans(
    question: str,
    model: str = 'gemini-3.1-flash-lite',
) -> List[dict]:
    """Call the LLM to extract and role-classify all temporal spans."""
    prompt = _build_extraction_prompt(question)
    raw = call_llm(prompt, model=model)
    return _parse_json_array(raw)


def select_chief_complaint_duration(spans: List[dict]) -> Optional[dict]:
    """From a span list, return the best representative of chief complaint duration.

    Priority:
      1. a span with role='chief_complaint' and a numeric `days` value
      2. a span with role='onset_to_presentation' and a numeric `days` value
      3. None
    If multiple candidates, prefer the one with largest `days` (usually the more
    temporally significant one; short ICU monitoring intervals are rejected via
    role='observation_window').
    """
    for roles in (['chief_complaint'], ['onset_to_presentation']):
        candidates = [s for s in spans if s['role'] in roles and s.get('days') is not None]
        if not candidates:
            continue
        return max(candidates, key=lambda s: s['days'])
    return None


def extract_patient_duration_llm(
    question: str,
    model: str = 'gemini-3.1-flash-lite',
) -> Dict:
    """Drop-in replacement signature for the regex version.

    Returns {value, unit, days, source='llm_role_tagged', spans, role}
    """
    spans = extract_patient_temporal_spans(question, model=model)
    chief = select_chief_complaint_duration(spans)
    if chief is None:
        return {
            'value': None, 'unit': None, 'days': None,
            'source': 'llm_role_tagged', 'role': None, 'spans': spans,
        }
    return {
        'value': chief.get('value'),
        'unit': chief.get('unit'),
        'days': chief.get('days'),
        'source': 'llm_role_tagged',
        'role': chief['role'],
        'span': chief.get('span'),
        'spans': spans,
    }


# ── CLI: batch-extract patient durations for a dataset ─────────────────

def batch_extract(dataset_path: str, output_path: str,
                  model: str = 'gemini-3.1-flash-lite',
                  limit: Optional[int] = None,
                  sleep_seconds: float = 0.3):
    """Batch-run LLM extraction on a dataset and cache results by uid."""
    import time
    with open(dataset_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get('results', data.get('data', data))

    out = {}
    for i, item in enumerate(data[:limit] if limit else data):
        uid = item.get('uid', item.get('idx', i))
        try:
            result = extract_patient_duration_llm(item['question'], model=model)
        except Exception as e:
            result = {'value': None, 'unit': None, 'days': None,
                      'source': 'llm_role_tagged', 'error': str(e)}
        out[uid] = result
        if (i + 1) % 10 == 0:
            n_extracted = sum(1 for r in out.values() if r.get('days') is not None)
            print(f'  [{i+1}/{len(data) if not limit else limit}] extracted={n_extracted}')
        time.sleep(sleep_seconds)

    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    n_ok = sum(1 for r in out.values() if r.get('days') is not None)
    print(f'\nExtracted chief_complaint duration for {n_ok}/{len(out)} items')
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cache/validated_cf_tc_83_1of3.json')
    parser.add_argument('--output', default='cache/patient_durations_llm_cf83.json')
    parser.add_argument('--model', default='gemini-3.1-flash-lite')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    batch_extract(args.dataset, args.output, args.model, args.limit)
