#!/usr/bin/env python3
"""Re-cut the reader results on the subsets verify_duration_critical.py established by experiment.

If duration-guided retrieval works because of duration, its margin should be larger on the
questions where duration is demonstrably load-bearing than on the ones where it demonstrably is
not. This slices the same frozen runs both ways instead of arguing from the aggregate.

Per-question accuracy is the mean over runs (0..1), so a question the reader gets right in 3 of 5
runs contributes 0.6 rather than being forced to right/wrong.

Usage:  DATASET=329 RESULTS_DIR=results/round2_intentfree python3 pipeline/results_by_verified_subset.py
"""
import json, os, statistics
from pathlib import Path

PIPE = Path(__file__).resolve().parent
DS = os.environ.get("DATASET", "329")
RES = PIPE / os.environ.get("RESULTS_DIR", "results/round2_intentfree")
MODEL = os.environ.get("MODEL", "gpt-5.4-mini")

subsets = json.load(open(PIPE / "verification/verified_subsets.json"))[DS]
GROUPS = [("duration_critical",     set(subsets["duration_critical"])),
          ("not_duration_critical", set(subsets["not_duration_critical"])),
          ("control_fail",          set(subsets["control_fail"])),
          ("ALL",                   None)]


def per_question(method):
    f = RES / f"{DS}_{method}_{MODEL.replace('/','_')}.json"
    if not f.exists(): return None
    runs = json.load(open(f))["runs"]
    acc = {}
    for run in runs:
        for r in run["results"]:
            acc.setdefault(r["uid"], []).append(1.0 if r["is_correct"] else 0.0)
    return {u: statistics.fmean(v) for u, v in acc.items()}


methods = [m for m in ["walker", "walker_interval", "raw_1hop", "raw_2hop", "hykge",
                       "vanilla", "cot", "medrag", "tog"] if (per_question(m) is not None)]
PQ = {m: per_question(m) for m in methods}

hdr = "".join(f"{g:>22s}" for g, _ in GROUPS)
print(f"{DS} · {MODEL} · accuracy % by verified subset\n")
print(f"{'method':16s}{hdr}")
print("-" * (16 + 22 * len(GROUPS)))
for m in methods:
    cells = ""
    for g, uids in GROUPS:
        sel = [v for u, v in PQ[m].items() if uids is None or u in uids]
        cells += f"{(f'{100*statistics.fmean(sel):.1f}  (n={len(sel)})' if sel else '—'):>22s}"
    print(f"{m:16s}{cells}")

if "walker" in methods:
    print(f"\nwalker minus each baseline (percentage points):")
    print(f"{'baseline':16s}{hdr}")
    print("-" * (16 + 22 * len(GROUPS)))
    for m in methods:
        if m == "walker": continue
        cells = ""
        for g, uids in GROUPS:
            sel = [u for u in PQ["walker"] if u in PQ[m] and (uids is None or u in uids)]
            if not sel: cells += f"{'—':>22s}"; continue
            d = 100 * (statistics.fmean(PQ["walker"][u] for u in sel)
                       - statistics.fmean(PQ[m][u] for u in sel))
            cells += f"{f'{d:+.1f}  (n={len(sel)})':>22s}"
        print(f"{m:16s}{cells}")
    print("\nNote: the duration_critical column is small by construction — read its sign, not its "
          "significance.")
