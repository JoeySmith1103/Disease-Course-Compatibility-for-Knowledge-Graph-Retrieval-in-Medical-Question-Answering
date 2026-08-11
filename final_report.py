#!/usr/bin/env python3
"""One consolidated report over every dataset that has reader results.

Per dataset it prints: per-run counts, mean ± std, parse failures, a paired test against the
reference method, and — where a hand review exists — accuracy on the temporal-critical subset.

The critical column is only interpretable next to a no-KG baseline. Every method rising on that
subset means the subset is easier, not that retrieval helped; the lift that matters is a method's
gain MINUS vanilla's gain on the same questions, so vanilla is printed as the reference line
whenever it is available and the "vs vanilla" column is computed from it.

Usage:  RESULTS_DIR=results/revised_prompt python3 pipeline/final_report.py [dataset ...]
"""
import json, os, glob, statistics, sys
from pathlib import Path

PIPE  = Path(__file__).resolve().parent
RES   = PIPE / os.environ.get("RESULTS_DIR", "results/revised_prompt")
MODEL = os.environ.get("MODEL", "gpt-5.4-mini")
REF   = os.environ.get("REF", "walker__revised")
DATASETS = sys.argv[1:] or ["329", "medbullets", "mmlu"]
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


def load(ds):
    perq, meta = {}, {}
    for f in sorted(glob.glob(str(RES / f"{ds}_*_{MODEL.replace('/','_')}.json"))):
        d = json.load(open(f))
        acc = {}
        for run in d["runs"]:
            for r in run["results"]:
                acc.setdefault(r["uid"], []).append(1.0 if r["is_correct"] else 0.0)
        perq[d["method"]] = {u: statistics.fmean(v) for u, v in acc.items()}
        meta[d["method"]] = d
    return perq, meta


def critical_uids(ds):
    p = PIPE / f"datasets/{ds}/temporal_critical.json"
    return set(json.load(open(p))["uids"]) if p.exists() else set()


for ds in DATASETS:
    perq, meta = load(ds)
    if not perq:
        print(f"\n### {ds}: 尚無結果\n"); continue
    crit = critical_uids(ds)
    order = sorted(meta, key=lambda m: -meta[m]["mean_acc"])
    w = max(len(m) for m in order) + 2
    n = meta[order[0]]["n"]

    print(f"\n{'='*len(f'### {ds} · {MODEL} · {RES.name} · n={n}')}")
    print(f"### {ds} · {MODEL} · {RES.name} · n={n}")
    print("=" * len(f"### {ds} · {MODEL} · {RES.name} · n={n}"))

    print(f"\n{'method':{w}}{'runs':>20}{'mean ± std':>18}{'unparseable':>14}")
    print("-" * (w + 52))
    for m in order:
        d = meta[m]; ma, sa = d['mean_acc'], d['std_acc']
        print(f"{m:{w}}{str(d['runs_correct']):>20}"
              f"{f'{ma:.2f} ± {sa:.2f}%':>18}"
              f"{str(d.get('runs_unparseable')):>14}")

    # paired vs REF, all questions
    if REF in perq:
        print(f"\n配對比較 vs {REF}（全部題目）")
        print(f"{'method':{w}}{'Δ pp':>9}{'win':>6}{'loss':>6}{'tie':>6}{'p':>10}")
        print("-" * (w + 37))
        for m in order:
            if m == REF: continue
            common = sorted(set(perq[REF]) & set(perq[m]))
            a = [perq[REF][u] for u in common]; b = [perq[m][u] for u in common]
            win = sum(1 for x, y in zip(a, b) if x > y); loss = sum(1 for x, y in zip(a, b) if x < y)
            p = "—"
            if wilcoxon and any(x != y for x, y in zip(a, b)):
                try: p = f"{wilcoxon(a, b).pvalue:.4f}"
                except Exception: pass
            print(f"{m:{w}}{100*(statistics.fmean(a)-statistics.fmean(b)):>+9.2f}"
                  f"{win:>6}{loss:>6}{len(common)-win-loss:>6}{p:>10}")

    if not crit:
        print(f"\n（{ds} 無逐題人工判讀，不做 temporal-critical 分層）")
        continue

    # ALL vs critical, with vanilla as the yardstick for "is the subset just easier?"
    base = "vanilla" if "vanilla" in perq else None
    base_lift = None
    if base:
        ba = 100 * statistics.fmean(perq[base].values())
        bc = 100 * statistics.fmean([v for u, v in perq[base].items() if u in crit])
        base_lift = bc - ba
    print(f"\ntemporal-critical 分層（{len(crit & set(perq[order[0]]))}/{n} 題）")
    hdr = f"{'method':{w}}{'ALL %':>9}{'critical %':>13}{'lift':>8}"
    if base_lift is not None: hdr += f"{'vs vanilla':>12}"
    print(hdr); print("-" * len(hdr))
    for m in order:
        a = 100 * statistics.fmean(perq[m].values())
        sel = [v for u, v in perq[m].items() if u in crit]
        if not sel:
            print(f"{m:{w}}{a:>9.2f}{'—':>13}{'—':>8}"); continue
        c = 100 * statistics.fmean(sel)
        line = f"{m:{w}}{a:>9.2f}{c:>13.2f}{c-a:>+8.2f}"
        if base_lift is not None: line += f"{(c-a)-base_lift:>+12.2f}"
        print(line)
    if base_lift is not None:
        print(f"\n  vanilla 在 critical 子集的 lift = {base_lift:+.2f}pp（無任何 KG）。")
        print(f"  「vs vanilla」欄扣掉這個基準 —— 那才是 KG 帶來的部分，")
        print(f"  正值才代表該方法在時間依賴題上真的多做了事。")
    else:
        print(f"\n  ⚠ 沒有 vanilla 結果，無法判斷 lift 是方法效果還是子集本身較易。")

    if REF in perq and crit:
        print(f"\ncritical 子集上 vs {REF}")
        print(f"{'method':{w}}{'Δ pp':>9}{'n':>6}{'win':>6}{'loss':>6}{'tie':>6}{'p':>10}")
        print("-" * (w + 43))
        for m in order:
            if m == REF: continue
            common = sorted(crit & set(perq[REF]) & set(perq[m]))
            if not common: continue
            a = [perq[REF][u] for u in common]; b = [perq[m][u] for u in common]
            win = sum(1 for x, y in zip(a, b) if x > y); loss = sum(1 for x, y in zip(a, b) if x < y)
            p = "—"
            if wilcoxon and any(x != y for x, y in zip(a, b)):
                try: p = f"{wilcoxon(a, b).pvalue:.4f}"
                except Exception: pass
            print(f"{m:{w}}{100*(statistics.fmean(a)-statistics.fmean(b)):>+9.2f}"
                  f"{len(common):>6}{win:>6}{loss:>6}{len(common)-win-loss:>6}{p:>10}")
        print("  子集題數有限 — 看方向，不要當顯著性用。")
print()
