#!/usr/bin/env python3
"""Freeze a KG-RAG baseline's retrieval output into frozen/<ds>/<method>.json.

The baseline retrieval scripts (hykge_baseline.py, tog_baseline.py, run_medrag_textbook.py)
write their results to cache/ with the retrieved evidence in `kg_block` and the assembled
prompt in `prompt_full`. This converts that into the SAME frozen schema every other method
uses — {uid, gold, route, kg_block, prompt} — so run_reader.py / run_reader_block.py can point
ANY model at the baseline without redoing retrieval.

(Previously no such step existed in pipeline/, which is why only walker had a build->freeze
path and the other baselines' frozen files had to be produced outside this folder.)

Usage:
  DATASET=1273 METHOD=hykge SRC=cache/hykge_baseline_1273__gpt-5.4-mini.json \
    python3 pipeline/freeze_baseline.py
"""
import json, os, re, sys
from pathlib import Path


def _lines(t):
    """Evidence lines with list markers stripped — the same normalisation run_reader's canary uses."""
    return {re.sub(r"^\s*(?:\d+\.|[-*])\s*", "", ln).strip()
            for ln in (t or "").splitlines() if ln.strip()}


def pick_evidence(r):
    """Choose the field that holds the evidence the prompt was actually built from.

    Baselines disagree on which field that is, and the name is not a reliable guide: hykge writes
    the block it renders, while tog wrote a scored ENTITY ranking into kg_block but rendered the
    CHAINS. Freezing the wrong field is invisible for as long as the frozen prompt is replayed
    verbatim — and then a re-render, which rebuilds the prompt FROM kg_block, quietly swaps the
    method's evidence for a different representation of it.

    So pick by containment, not by field name: the frozen kg_block is whichever candidate the
    prompt demonstrably contains. Returns (field_name, text).
    """
    prompt = r.get("prompt_full") or r.get("prompt") or ""
    pl = _lines(prompt)
    cands = []
    for k in ("kg_block", "chains_full", "chains", "evidence_block", "entity_ranking"):
        v = r.get(k)
        if isinstance(v, list):
            v = "\n".join(f"  {i+1}. {x}" for i, x in enumerate(v))
        if isinstance(v, str) and v.strip():
            cands.append((k, v))
    for k, v in cands:
        if _lines(v) <= pl:
            return k, v
    return ("kg_block", r.get("kg_block") or "")

PIPE = Path(__file__).resolve().parent
DS = os.environ.get("DATASET", "1273")
METHOD = os.environ.get("METHOD")
SRC = os.environ.get("SRC")
if not METHOD or not SRC:
    sys.exit("need METHOD=<name> and SRC=<path to baseline cache json>")

src = Path(SRC)
if not src.is_absolute():
    src = PIPE / src if (PIPE / src).exists() else Path(SRC)
recs = json.load(open(src))
recs = recs.get("results", recs)

bench = {x["uid"]: x for x in json.load(open(PIPE / f"datasets/{DS}/benchmark.json"))}

items, missing_prompt, missing_kg, unmatched = [], 0, 0, 0
picked = {}
for r in recs:
    uid = r.get("uid")
    if uid not in bench:
        continue
    prompt = r.get("prompt_full") or r.get("prompt") or ""
    if not prompt:
        missing_prompt += 1
        continue
    field, kg = pick_evidence(r)
    picked[field] = picked.get(field, 0) + 1
    if not kg:
        missing_kg += 1
    # Same canary as the readers: evidence must actually be inside the prompt we froze. Compared
    # by line CONTENT, not verbatim substring — some baselines number the block ("  1. ") and
    # bullet it in the prompt ("- "), which is formatting, not evidence loss.
    elif not (_lines(kg) <= _lines(prompt)):
        unmatched += 1
    items.append({"uid": uid, "gold": r.get("gold", bench[uid].get("answer")),
                  "route": r.get("route", METHOD), "kg_block": kg, "prompt": prompt})

out = PIPE / f"frozen/{DS}/{METHOD}.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump({"method": METHOD, "dataset": DS, "n": len(items), "items": items},
          open(out, "w"), indent=1)
print(f"froze {len(items)}/{len(bench)} -> {out}")
print(f"  evidence field used: " + ", ".join(f"{k}×{v}" for k, v in sorted(picked.items(), key=lambda t: -t[1])))
if missing_prompt: print(f"  skipped {missing_prompt} record(s) with no stored prompt")
if missing_kg:     print(f"  {missing_kg} record(s) had an empty kg_block")
if unmatched:
    # loud, because re-rendering from a kg_block the prompt never contained produces a prompt
    # variant that is not the same method any more
    print(f"  ⚠ {unmatched}/{len(items)} record(s): NO stored field matches the prompt — "
          f"re-rendered variants of {METHOD} would carry evidence the original prompt never had")
