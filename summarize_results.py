#!/usr/bin/env python3
"""Summarise reader results for one dataset: mean ± std, parse failures, and paired tests.

Reports unparseable alongside accuracy because an unparsed answer is scored WRONG — a method that
merely formats badly looks like a method that reasons badly, and the two need to be told apart
before any number is quoted.

Paired comparison is against the reference method on the questions BOTH methods answered, using
each question's mean-over-runs score (0..1) rather than a single run, so run-to-run noise does not
drive the test. Wilcoxon signed-rank; p-values are raw (apply your own multiple-comparison
correction when quoting several at once).

Usage:
  DATASET=329 RESULTS_DIR=results/revised_prompt REF=walker__revised \
    python3 pipeline/summarize_results.py
"""
import json, os, statistics, glob, re
from pathlib import Path

PIPE = Path(__file__).resolve().parent
DS   = os.environ.get("DATASET", "329")
RES  = PIPE / os.environ.get("RESULTS_DIR", "results/revised_prompt")
MODEL = os.environ.get("MODEL", "gpt-5.4-mini")
REF  = os.environ.get("REF", "walker__revised")

rows, perq = [], {}
for f in sorted(glob.glob(str(RES / f"{DS}_*_{MODEL.replace('/','_')}.json"))):
    d = json.load(open(f))
    m = d["method"]
    rows.append((m, d["n"], d["mean_acc"], d["std_acc"], d.get("runs_correct"),
                 d.get("runs_unparseable"), d.get("mean_unparseable", 0)))
    acc = {}
    for run in d["runs"]:
        for r in run["results"]:
            acc.setdefault(r["uid"], []).append(1.0 if r["is_correct"] else 0.0)
    perq[m] = {u: statistics.fmean(v) for u, v in acc.items()}

if not rows:
    raise SystemExit(f"no result files in {RES} for dataset {DS}")

rows.sort(key=lambda r: -r[2])
w = max(len(r[0]) for r in rows) + 2
print(f"\n{DS} · {MODEL} · {RES.name}\n")
print(f"{'method':{w}}{'n':>6}{'acc %':>16}{'runs':>22}{'unparseable':>16}")
print("-" * (w + 60))
for m, n, acc, sd, runs, unp, munp in rows:
    print(f"{m:{w}}{n:>6}{acc:>10.2f} ± {sd:<4.2f}{str(runs):>22}"
          f"{str(unp) + f' ({100*munp/n:.2f}%)':>16}")

if REF in perq:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    print(f"\npaired vs {REF} (per-question mean over runs, common questions only)")
    print(f"{'method':{w}}{'Δ pp':>9}{'n':>7}{'win':>6}{'loss':>6}{'tie':>6}{'p':>10}")
    print("-" * (w + 44))
    for m in [r[0] for r in rows]:
        if m == REF: continue
        common = sorted(set(perq[REF]) & set(perq[m]))
        a = [perq[REF][u] for u in common]; b = [perq[m][u] for u in common]
        win  = sum(1 for x, y in zip(a, b) if x > y)
        loss = sum(1 for x, y in zip(a, b) if x < y)
        tie  = len(common) - win - loss
        d = 100 * (statistics.fmean(a) - statistics.fmean(b))
        p = "—"
        if wilcoxon and any(x != y for x, y in zip(a, b)):
            try: p = f"{wilcoxon(a, b).pvalue:.4f}"
            except Exception: pass
        print(f"{m:{w}}{d:>+9.2f}{len(common):>7}{win:>6}{loss:>6}{tie:>6}{p:>10}")

# accuracy restricted to the duration-critical subset, when one has been established
sub = PIPE / "verification/verified_subsets.json"
hand = PIPE / f"verification/manual_read_{DS}.jsonl"
crit = set()
if hand.exists():
    crit = {json.loads(l)["uid"] for l in open(hand) if json.loads(l)["verdict"] == "critical"}
    src = "hand review"
elif sub.exists():
    j = json.load(open(sub))
    if DS in j: crit = set(j[DS]["duration_critical"]); src = "script"
if crit:
    print(f"\nduration-critical subset ({len(crit)} 題, {src})")
    print(f"{'method':{w}}{'all %':>9}{'critical %':>13}")
    print("-" * (w + 22))
    for m, *_ in rows:
        sel = [v for u, v in perq[m].items() if u in crit]
        allv = list(perq[m].values())
        c = f"{100*statistics.fmean(sel):.2f}" if sel else "—"
        print(f"{m:{w}}{100*statistics.fmean(allv):>9.2f}{c:>13}")
    print("  注意：critical 子集題數少，看方向不看顯著性。")
