#!/usr/bin/env python3
"""Old prompt vs revised prompt, same frozen evidence, per method.

This is a controlled comparison in the strict sense: <method>.json and <method>__revised.json are
built from a kg_block that is byte-identical, so the only thing that differs between the two
columns is wording. Any gap is a prompt effect.

The number that matters is not which prompt scores higher on average but how far apart the methods
sit under each one. A template that lifts everything equally is harmless; a template that squeezes
six methods into a narrow band has destroyed the contrast the experiment exists to measure — so the
spread (max − min) is printed under each column, and that is the headline.

Paired per-question tests use the mean-over-runs accuracy, so a question answered right in 2 of 3
runs counts 0.67 instead of being forced to a single run's verdict.

Usage:  python3 pipeline/prompt_ab_report.py [dataset ...]      # default: medbullets mmlu
"""
import json, os, statistics, sys
from pathlib import Path

PIPE  = Path(__file__).resolve().parent
MODEL = os.environ.get("MODEL", "gpt-5.4-mini")
OLD   = PIPE / "results/old_prompt"
NEW   = PIPE / "results/revised_prompt"
METHODS = ["walker", "walker_interval", "raw_1hop", "raw_2hop", "tog", "hykge"]
DATASETS = sys.argv[1:] or ["medbullets", "mmlu"]
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


def load(d, ds, method):
    f = d / f"{ds}_{method}_{MODEL.replace('/', '_')}.json"
    if not f.exists():
        return None
    j = json.load(open(f))
    acc = {}
    for run in j["runs"]:
        for r in run["results"]:
            acc.setdefault(r["uid"], []).append(1.0 if r["is_correct"] else 0.0)
    return {"meta": j, "perq": {u: statistics.fmean(v) for u, v in acc.items()},
            "n_runs": len(j["runs"])}


for ds in DATASETS:
    rows = []
    for m in METHODS:
        o, n = load(OLD, ds, m), load(NEW, ds, m + "__revised")
        if o or n:
            rows.append((m, o, n))
    if not rows:
        print(f"\n### {ds}: 尚無資料\n"); continue

    w = max(len(m) for m, _, _ in rows) + 2
    print(f"\n{'='*68}\n### {ds} · {MODEL} · 舊 prompt vs revised（證據完全相同）\n{'='*68}")
    # CJK glyphs are double-width in a terminal but count as one character to str.format, so a
    # column padded by len() drifts; pad by display width instead or the table will not line up
    def cell(s, width):
        pad = width - sum(2 if ord(c) > 0x2E80 else 1 for c in str(s))
        return " " * max(pad, 1) + str(s)

    print(f"\n{'method':{w}}{cell('舊 prompt', 20)}{cell('revised', 20)}{cell('Δ (舊−新)', 14)}{cell('p', 10)}")
    print("-" * (w + 64))
    olds, news = [], []
    for m, o, n in rows:
        so = f"{o['meta']['mean_acc']:.2f} ± {o['meta']['std_acc']:.2f} (N={o['n_runs']})" if o else "—"
        sn = f"{n['meta']['mean_acc']:.2f} ± {n['meta']['std_acc']:.2f} (N={n['n_runs']})" if n else "—"
        d = p = "—"
        if o and n:
            olds.append(o['meta']['mean_acc']); news.append(n['meta']['mean_acc'])
            d = f"{o['meta']['mean_acc'] - n['meta']['mean_acc']:+.2f}"
            common = sorted(set(o["perq"]) & set(n["perq"]))
            a = [o["perq"][u] for u in common]; b = [n["perq"][u] for u in common]
            if wilcoxon and any(x != y for x, y in zip(a, b)):
                try: p = f"{wilcoxon(a, b).pvalue:.4f}"
                except Exception: pass
        print(f"{m:{w}}{cell(so, 20)}{cell(sn, 20)}{cell(d, 14)}{cell(p, 10)}")

    if len(olds) >= 2 and len(news) >= 2:
        print("-" * (w + 64))
        print(f"{cell('全距 (max−min)', w)}{cell(f'{max(olds)-min(olds):.2f}', 20)}"
              f"{cell(f'{max(news)-min(news):.2f}', 20)}")
        print(f"{cell('平均', w)}{cell(f'{statistics.fmean(olds):.2f}', 20)}"
              f"{cell(f'{statistics.fmean(news):.2f}', 20)}"
              f"{cell(f'{statistics.fmean(olds)-statistics.fmean(news):+.2f}', 14)}")
        print(f"\n  全距才是重點：整體高低可以靠 prompt 補，方法之間被壓平就補不回來。")
        print(f"  舊 prompt 全距 {max(olds)-min(olds):.2f}pp，revised {max(news)-min(news):.2f}pp。")
        if any(r[1] and r[1]['n_runs'] == 1 for r in rows):
            print(f"  ⚠ 有方法只有 N=1，單輪噪音約 ±1pp，Δ 小於 1pp 的不要當結論。")
print()
