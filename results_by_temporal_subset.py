#!/usr/bin/env python3
"""Reader accuracy on all questions vs on the temporal-critical ones only.

Aggregate accuracy cannot show the claim the thesis makes. A method that helps only where the
answer depends on time looks identical to one that helps everywhere until the questions are split
by that property — so this prints two columns, ALL and critical, plus a paired test on the
critical subset.

The critical labels are read from datasets/<ds>/temporal_critical.json, written by
label_temporal_critical.py from the per-question hand review — not from a keyword rule. 329 is not meant to be sliced this
way: it was assembled to be duration-critical in the first place, so every question is already in
the subset by construction.

Per-question accuracy is the mean over runs (0..1), so a question answered right in 2 of 3 runs
counts 0.67 rather than being forced to right/wrong by whichever run is looked at.

Usage:
  DATASET=medbullets RESULTS_DIR=results/revised_prompt REF=walker__revised \
    python3 pipeline/results_by_temporal_subset.py
"""
import json, os, glob, statistics
from pathlib import Path

PIPE  = Path(__file__).resolve().parent
DS    = os.environ.get("DATASET", "medbullets")
RES   = PIPE / os.environ.get("RESULTS_DIR", "results/revised_prompt")
MODEL = os.environ.get("MODEL", "gpt-5.4-mini")
REF   = os.environ.get("REF", "")

# labels live on the dataset now (written by label_temporal_critical.py), so no join is needed
tc = PIPE / f"datasets/{DS}/temporal_critical.json"
if not tc.exists():
    raise SystemExit(f"no temporal labels for {DS}\n"
                     f"  run: python3 pipeline/label_temporal_critical.py {DS}")
crit = set(json.load(open(tc))["uids"])

perq, order = {}, []
for f in sorted(glob.glob(str(RES / f"{DS}_*_{MODEL.replace('/','_')}.json"))):
    d = json.load(open(f))
    acc = {}
    for run in d["runs"]:
        for r in run["results"]:
            acc.setdefault(r["uid"], []).append(1.0 if r["is_correct"] else 0.0)
    perq[d["method"]] = {u: statistics.fmean(v) for u, v in acc.items()}
    order.append((d["method"], d["mean_acc"]))
if not perq:
    raise SystemExit(f"no result files in {RES} for {DS}")
order.sort(key=lambda t: -t[1])
methods = [m for m, _ in order]
w = max(len(m) for m in methods) + 2
allu = set(perq[methods[0]])

print(f"\n{DS} · {MODEL} · {RES.name}")
print(f"critical: {len(crit & allu)}/{len(allu)} 題（datasets/{DS}/temporal_critical.json）\n")
print(f"{'method':{w}}{'ALL %':>10}{'critical %':>14}{'Δ':>9}")
print("-" * (w + 33))
for m in methods:
    a = 100 * statistics.fmean(perq[m].values())
    sel = [v for u, v in perq[m].items() if u in crit]
    if sel:
        c = 100 * statistics.fmean(sel)
        print(f"{m:{w}}{a:>10.2f}{c:>14.2f}{c-a:>+9.2f}")
    else:
        print(f"{m:{w}}{a:>10.2f}{'—':>14}{'—':>9}")

if REF and REF in perq:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    print(f"\ncritical 子集上，各方法 vs {REF}")
    print(f"{'method':{w}}{'Δ pp':>9}{'n':>6}{'win':>6}{'loss':>6}{'tie':>6}{'p':>10}")
    print("-" * (w + 43))
    for m in methods:
        if m == REF: continue
        common = sorted(crit & set(perq[REF]) & set(perq[m]))
        if not common: continue
        a = [perq[REF][u] for u in common]; b = [perq[m][u] for u in common]
        win  = sum(1 for x, y in zip(a, b) if x > y)
        loss = sum(1 for x, y in zip(a, b) if x < y)
        p = "—"
        if wilcoxon and any(x != y for x, y in zip(a, b)):
            try: p = f"{wilcoxon(a, b).pvalue:.4f}"
            except Exception: pass
        print(f"{m:{w}}{100*(statistics.fmean(a)-statistics.fmean(b)):>+9.2f}"
              f"{len(common):>6}{win:>6}{loss:>6}{len(common)-win-loss:>6}{p:>10}")
    print("  子集題數有限 — 看方向，不要當顯著性用。")

# the results dir belongs in the name: running this for old_prompt and then revised_prompt used to
# write the same file twice, so the second silently replaced the first
out = PIPE / f"results/_temporal_subset_{DS}_{RES.name}.json"
out.parent.mkdir(exist_ok=True)
json.dump({"dataset": DS, "results_dir": RES.name, "critical_uids": sorted(crit & allu),
           "accuracy": {m: {"all": 100 * statistics.fmean(perq[m].values()),
                            "critical": (100 * statistics.fmean([v for u, v in perq[m].items() if u in crit])
                                         if crit & set(perq[m]) else None)}
                        for m in methods}}, open(out, "w"), indent=1)
print(f"\nsaved -> {out}")
