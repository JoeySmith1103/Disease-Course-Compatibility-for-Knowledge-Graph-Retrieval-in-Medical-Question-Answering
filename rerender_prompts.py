#!/usr/bin/env python3
"""Re-render the frozen prompts with a DIFFERENT template, reusing the SAME kg_block.

Prompt tuning does NOT need retrieval. `frozen/<ds>/<method>.json` stores `kg_block` separately
from `prompt`, so a new prompt wording can be rendered straight from the existing retrieval —
no Neo4j, no LLM calls, runs in seconds. Only the wording changes; the evidence is identical, so
prompt variants are a clean controlled comparison.

Two ways to supply the new wording:
  1. TEMPLATE=<name>      — any template in prompts.py (WALKER, RAW_KG, MEDRAG, NO_KG, ...)
  2. TEMPLATE_FILE=<path> — a .txt file you wrote yourself

The template may use: {question} {options_block} {kg_block} {patient_dur_str} {symptom_duration}
(A template that omits {kg_block} is allowed — that is how you test "same questions, no evidence".)

Usage:
  DATASET=1273 METHOD=walker VARIANT=myv1 TEMPLATE_FILE=my_prompt.txt \
    python3 rerender_prompts.py
  # → frozen/1273/walker__myv1.json     (then: METHOD=walker__myv1 python3 run_reader.py)
"""
import json, os, re, sys, importlib.util
from pathlib import Path

PIPE = Path(__file__).resolve().parent
DS      = os.environ.get("DATASET", "1273")
METHOD  = os.environ.get("METHOD", "walker")
VARIANT = os.environ.get("VARIANT")
TEMPLATE      = os.environ.get("TEMPLATE")
TEMPLATE_FILE = os.environ.get("TEMPLATE_FILE")
if not VARIANT:
    sys.exit("need VARIANT=<name> (output goes to frozen/<ds>/<method>__<variant>.json)")

_spec = importlib.util.spec_from_file_location("P", PIPE / "prompts.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)

if TEMPLATE_FILE:
    tmpl = Path(TEMPLATE_FILE).read_text()
    src_desc = f"file:{TEMPLATE_FILE}"
elif TEMPLATE:
    tmpl = getattr(P, TEMPLATE)
    src_desc = f"prompts.{TEMPLATE}"
else:
    sys.exit("need TEMPLATE=<name in prompts.py> or TEMPLATE_FILE=<path>")

DD = PIPE / f"datasets/{DS}"
bench = {x["uid"]: x for x in json.load(open(DD / "benchmark.json"))}
dur_c = json.load(open(DD / "durations.json"))
items = json.load(open(PIPE / f"frozen/{DS}/{METHOD}.json"))["items"]


def days_to_phrase(d):
    if isinstance(d, (list, tuple)):
        seen = []
        for x in d:
            p = days_to_phrase(x)
            if p not in seen: seen.append(p)
        return " / ".join(seen) if len(seen) > 1 else seen[0]
    if d < 1:   return f"{int(d*24)} hours"
    if d < 14:  return f"{int(d)} days"
    if d < 60:  return f"{int(d/7)} weeks"
    if d < 365: return f"{int(d/30)} months"
    return f"{int(d/365)} years"


def ob(o):
    return "\n".join(f"  {k}. {v}" for k, v in sorted(o.items()))



def duration_to_hours_text(text):
    raw = (text or "").strip()
    if not raw:
        return "not stated"
    low = raw.lower()
    if "not stated" in low or "not applicable" in low:
        return "not stated"
    unit_re = r"seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?"
    range_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:-|to|–)\s*([0-9]+(?:\.[0-9]+)?)\s*(" + unit_re + r")", low)
    if range_m:
        value = (float(range_m.group(1)) + float(range_m.group(2))) / 2.0
        unit = range_m.group(3)
    else:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(" + unit_re + r")", low)
        if not m:
            return "not stated"
        value = float(m.group(1))
        unit = m.group(2)
    if unit.startswith(("second", "sec")):
        hours = value / 3600.0
    elif unit.startswith(("minute", "min")):
        hours = value / 60.0
    elif unit.startswith(("hour", "hr")):
        hours = value
    elif unit.startswith("day"):
        hours = value * 24.0
    elif unit.startswith("week"):
        hours = value * 24.0 * 7.0
    elif unit.startswith("month"):
        hours = value * 24.0 * 30.0
    elif unit.startswith("year"):
        hours = value * 24.0 * 365.0
    else:
        return "not stated"
    return str(int(round(hours))) if abs(hours - round(hours)) < 1e-9 else f"{hours:.2f}".rstrip("0").rstrip(".")

def duration_str(it):
    """Prefer the duration the ORIGINAL prompt was frozen with (keeps multi-axis strings like
    '2 days / 7 days' exact); fall back to durations.json."""
    m = re.search(r"Patient symptom duration:\s*(.+)", it.get("prompt") or "")
    if m:
        return m.group(1).strip()
    d = (dur_c.get(it["uid"]) or {}).get("days")
    return days_to_phrase(d) if isinstance(d, (int, float)) and d > 0 else P.NO_DURATION_STR


# Readable aliases for the template slots. A template written by hand is far more likely to say
# {options} than {options_block}, and silently dropping an unrecognised name would produce a
# prompt with a literal "{options}" in it — so unknown names are a hard error below, not a skip.
ALIAS = {"options": "options_block", "option_block": "options_block",
         "patient_duration": "patient_dur_str", "duration": "patient_dur_str",
         "retrieved_information": "kg_block", "retrieval": "kg_block", "evidence": "kg_block",
         "kg": "kg_block", "duration_hours": "symptom_duration"}

fields = set(re.findall(r"\{(\w+)\}", tmpl))
CANON = {"question", "options_block", "kg_block", "patient_dur_str", "symptom_duration"}
unknown = {f for f in fields if f not in CANON and f not in ALIAS}
if unknown:
    sys.exit(f"template has unknown placeholder(s): {sorted(unknown)}\n"
             f"  supported: {sorted(CANON)}\n  aliases:   {sorted(ALIAS)}")

def strip_scores(block: str) -> str:
    """Drop the numeric score annotation, keep the rank, the [category] tag and the concept name.

    Only the trailing parenthesised score group is removed, never a parenthesis that belongs to the
    concept itself — UMLS names routinely end in '(disorder)' / '(qualifier value)', and a greedy
    rule would silently amputate them. Formats differ per method (walker: 'score=..: cos=..+bc=..',
    ToG: 'score=..', raw hops and HyKGE: none), so the pattern anchors on 'score=' and end-of-line."""
    return "\n".join(re.sub(r"\s*\(score=[^()]*\)\s*$", "", ln).rstrip()
                     for ln in block.split("\n"))


KG_TRANSFORM = os.environ.get("KG_TRANSFORM", "")   # "" | strip_scores
TRANSFORMS = {"strip_scores": strip_scores}
if KG_TRANSFORM and KG_TRANSFORM not in TRANSFORMS:
    sys.exit(f"unknown KG_TRANSFORM={KG_TRANSFORM!r}; available: {sorted(TRANSFORMS)}")
xform = TRANSFORMS.get(KG_TRANSFORM, lambda b: b)

out, n_kg, n_changed = [], 0, 0
kg_field = {f for f in fields if (ALIAS.get(f, f)) == "kg_block"}
for it in items:
    q = bench[it["uid"]]
    kg_raw = it.get("kg_block") or ""
    kg = xform(kg_raw)
    if kg != kg_raw: n_changed += 1
    dur_txt = duration_str(it)
    vals = {"question": q["question"], "options_block": ob(q["options"]),
            "kg_block": kg, "patient_dur_str": dur_txt,
            "symptom_duration": duration_to_hours_text(dur_txt)}
    prompt = tmpl.format(**{f: vals[ALIAS.get(f, f)] for f in fields})
    if kg and kg_field:
        assert kg in prompt, f"BUG: kg_block missing from prompt for {it['uid']}"
        n_kg += 1
    out.append({**it, "kg_block": kg, "prompt": prompt})

dst = PIPE / f"frozen/{DS}/{METHOD}__{VARIANT}.json"
json.dump({"method": f"{METHOD}__{VARIANT}", "dataset": DS, "n": len(out),
           "rerendered_from": METHOD, "template": src_desc, "items": out},
          open(dst, "w"), indent=1)
print(f"re-rendered {len(out)} prompts from '{METHOD}' using {src_desc}")
print(f"  placeholders used : {sorted(fields)}")
print(f"  kg_block injected : {n_kg}/{len(out)}")
print(f"  kg transform      : {KG_TRANSFORM or 'none'}"
      + (f"  (changed {n_changed}/{len(out)} blocks)" if KG_TRANSFORM else "   (evidence unchanged — only the wording differs)"))
print(f"  -> {dst}")
print(f"  now: DATASET={DS} METHOD={METHOD}__{VARIANT} MODEL=<model> python3 run_reader.py")
