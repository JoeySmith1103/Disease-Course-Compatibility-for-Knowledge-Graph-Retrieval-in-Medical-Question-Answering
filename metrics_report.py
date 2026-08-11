#!/usr/bin/env python3
"""Accuracy + precision / recall / F1 for every stored reader result.

Implements the definitions from new_metrics.py, with the four corrections described in metrics.py:
a fixed label set taken from the dataset's options (so methods share a denominator), both macros
averaged over the SAME classes with zero_division=0 (so F1 is meaningful and precision is not
inflated by silently dropping never-predicted letters), support reported alongside, and a
support-weighted average for label sets too imbalanced for a plain macro.

Reads the stored `predicted` verdicts — it recomputes nothing and rewrites no result file.

Usage:
  python3 pipeline/metrics_report.py                    # every dataset, every results dir
  python3 pipeline/metrics_report.py medbullets         # one dataset
  DETAIL=1 python3 pipeline/metrics_report.py 329       # + per-class table
"""
import json, glob, os, statistics, sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))
from metrics import compute_metrics, aggregate, _labels_from_options

MODEL  = os.environ.get("MODEL", "gpt-5.4-mini")
DETAIL = os.environ.get("DETAIL")
DATASETS = sys.argv[1:] or ["329", "medbullets", "mmlu"]

for ds in DATASETS:
    bpath = PIPE / f"datasets/{ds}/benchmark.json"
    if not bpath.exists():
        continue
    bench = json.load(open(bpath))
    labels = _labels_from_options(bench)
    gold_dist = {}
    for b in bench:
        gold_dist[b["answer"]] = gold_dist.get(b["answer"], 0) + 1

    for sub in ["old_prompt", "revised_prompt", ""]:
        pat = str(PIPE / "results" / sub / f"{ds}_*_{MODEL.replace('/','_')}.json")
        files = [f for f in sorted(glob.glob(pat)) if "/results/" in f]
        if sub == "":                       # skip the loose top-level dir, it holds older rounds
            continue
        if not files:
            continue

        rows = []
        for f in files:
            d = json.load(open(f))
            if "runs" not in d:
                continue
            rm = [compute_metrics(r["results"], labels=labels) for r in d["runs"]]
            rows.append((d["method"].replace("__revised", ""), aggregate(rm), rm))
        if not rows:
            continue
        rows.sort(key=lambda t: -t[1]["accuracy"])
        a0 = rows[0][1]

        print(f"\n{'='*94}")
        print(f"### {ds} · {sub} · {MODEL}   n={a0['n']}  runs={a0['n_runs']}  "
              f"classes={a0['n_classes']} (最小 support {a0['min_support']})")
        print("=" * 94)
        hdr = (f"{'method':17}{'Accuracy':>16}{'MacroP':>16}{'MacroR':>16}"
               f"{'ParseableP':>16}{'MacroF1':>10}{'WeightF1':>10}{'unparse':>12}")
        print(hdr); print("-" * len(hdr))
        for m, a, _ in rows:
            cell = lambda k: f"{a[k]:.2f} ± {a[k+'_std']:.2f}"
            print(f"{m:17}{cell('accuracy'):>16}{cell('macro_precision'):>16}"
                  f"{cell('macro_recall'):>16}{cell('parseable_precision'):>16}"
                  f"{a['macro_f1']:>10.2f}{a['weighted_f1']:>10.2f}"
                  f"{str(a['unparseable']):>12}")
        if a0["min_support"] < 5:
            print(f"\n  ⚠ 最小類別只有 {a0['min_support']} 題 — macro_* 由極少數題目主導，改看 WeightF1。")
            print(f"    gold 字母分布: {gold_dist}")

        if DETAIL:
            for m, _, rm in rows:
                pc = rm[0]["per_class"]
                print(f"\n  [{m}] run 1 per-class")
                print(f"    {'letter':8}{'support':>9}{'n_pred':>8}{'P':>8}{'R':>8}{'F1':>8}")
                for lab in labels:
                    c = pc.get(lab)
                    if not c or not c["support"]:
                        continue
                    print(f"    {lab:8}{c['support']:>9}{c['n_pred']:>8}"
                          f"{c['precision']:>8.1f}{c['recall']:>8.1f}{c['f1']:>8.1f}")
print()
