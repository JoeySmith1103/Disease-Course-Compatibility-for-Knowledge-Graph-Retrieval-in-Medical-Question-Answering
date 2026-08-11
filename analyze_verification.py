#!/usr/bin/env python3
"""Summarise verify_duration_critical.py output, and — for 329 — cross-tabulate the experimental
verdict against the labels the benchmark was built with.

The cross-tab is the point. `verdict` (truly_tc / borderline_tc) and the 46 `human_cf` items were
assigned by JUDGEMENT ("does duration look load-bearing here?"). This script asks whether that
judgement survives a COUNTERFACTUAL ("does the answer actually change when the duration changes?").
Where the two disagree, the judgement labels are the ones without evidence behind them.

Usage:  python3 pipeline/analyze_verification.py [dataset ...]     # default: all found
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

PIPE = Path(__file__).resolve().parent
VDIR = PIPE / "verification"
ORDER = ["duration_critical", "not_duration_critical", "ambiguous", "control_fail", "perturb_fail"]

names = sys.argv[1:] or sorted(p.name.split("_duration_critical_")[0]
                               for p in VDIR.glob("*_duration_critical_*.json"))

for ds in names:
    f = next(VDIR.glob(f"{ds}_duration_critical_*.json"), None)
    if not f: print(f"[{ds}] no verification file"); continue
    d = json.load(open(f)); R = d["results"]
    c = Counter(r["verdict"] for r in R)
    tested = c["duration_critical"] + c["not_duration_critical"]

    print("=" * 78)
    print(f"{ds}  (n={len(R)}, model={d['model']}, N={d['n_samples']} samples/question, "
          f"thresholds ≥{d['thresholds']['control_gold_hits_min']} / "
          f"≤{d['thresholds']['perturbed_gold_hits_max']})")
    for k in ORDER:
        print(f"  {k:24s} {c[k]:5d}  {100*c[k]/len(R):5.1f}%")
    if tested:
        print(f"  → duration-critical rate among CONCLUSIVE questions: "
              f"{c['duration_critical']}/{tested} = {100*c['duration_critical']/tested:.1f}%")
    print(f"  → duration-critical as a share of the whole set: "
          f"{c['duration_critical']}/{len(R)} = {100*c['duration_critical']/len(R):.1f}%")

    # cross-tab against the labels the benchmark shipped with
    bpath = PIPE / f"datasets/{ds}/benchmark.json"
    if not bpath.exists(): continue
    B = {b["uid"]: b for b in json.load(open(bpath))}
    for field in ("verdict", "audit_source"):
        if not any(B.get(r["uid"], {}).get(field) for r in R): continue
        tab = defaultdict(Counter)
        for r in R:
            tab[B.get(r["uid"], {}).get(field)][r["verdict"]] += 1
        print(f"\n  prior label `{field}` vs experiment:")
        print(f"    {'':22s}{'critical':>9s}{'not':>8s}{'ambig':>7s}{'ctrl_f':>8s}{'pert_f':>8s}"
              f"{'  rate(conclusive)':>18s}")
        for lab, cc in sorted(tab.items(), key=lambda x: -sum(x[1].values())):
            t = cc["duration_critical"] + cc["not_duration_critical"]
            rate = f"{100*cc['duration_critical']/t:.1f}% (n={t})" if t else "—"
            print(f"    {str(lab):22s}{cc['duration_critical']:9d}{cc['not_duration_critical']:8d}"
                  f"{cc['ambiguous']:7d}{cc['control_fail']:8d}{cc['perturb_fail']:8d}{rate:>18s}")
    print()

    # Threshold sensitivity — is the low rate a property of MedQA, or of a strict cut-off?
    # Recomputed from the stored per-sample answers, so this costs no extra calls. Only questions
    # that got as far as the perturbed vote can be re-scored; the rest are structurally excluded.
    print("  threshold sensitivity (recomputed from stored samples):")
    print(f"    {'control ≥':>10s}{'perturb ≤':>11s}{'critical':>10s}{'conclusive':>12s}{'rate':>9s}")
    scored = [r for r in R if "perturbed_samples" in r]
    N = d["n_samples"]
    for hi, lo in [(N, 0), (4, 1), (3, 2), (3, 1)]:
        crit = sum(1 for r in scored
                   if r["control_samples"].count(r["gold"]) >= hi
                   and r["perturbed_samples"].count(r["gold"]) <= lo)
        conc = sum(1 for r in scored
                   if r["control_samples"].count(r["gold"]) >= hi
                   and (r["perturbed_samples"].count(r["gold"]) <= lo
                        or r["perturbed_samples"].count(r["gold"]) >= hi))
        print(f"    {hi:>10d}{lo:>11d}{crit:>10d}{conc:>12d}"
              f"{(f'{100*crit/conc:.1f}%' if conc else '—'):>9s}")

    # what the flips land on — a flip to a clinically adjacent option is the interesting case
    flips = Counter(r.get("perturbed_answer") for r in R if r["verdict"] == "duration_critical")
    if flips: print(f"  flips land on: {dict(flips)}")
    print()

# the verified subsets are what downstream result tables should be re-cut on
sub = {}
for ds in names:
    f = next(VDIR.glob(f"{ds}_duration_critical_*.json"), None)
    if not f: continue
    R = json.load(open(f))["results"]
    sub[ds] = {v: sorted(r["uid"] for r in R if r["verdict"] == v) for v in ORDER}
json.dump(sub, open(VDIR / "verified_subsets.json", "w"), indent=1)
print(f"saved uid lists -> {VDIR/'verified_subsets.json'}")
