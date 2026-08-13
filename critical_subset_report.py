#!/usr/bin/env python3
"""Metrics on the hand-reviewed temporal-critical subset, next to the full set.

The claim the thesis makes is conditional — duration-guided retrieval should help where the answer
depends on duration — so a method that helps everywhere and a method that helps only on time-
dependent questions look identical until the questions are split by that property.

Reading the critical column ALONE is a trap: every method rises on it, which usually means the
subset is easier rather than that retrieval worked. vanilla is therefore printed first and its own
rise is subtracted — "vs van" is a method's lift minus vanilla's lift, i.e. what the knowledge
graph contributed over a reader with no retrieval at all. Only a positive value there is evidence
for the claim.

Labels come from datasets/<ds>/temporal_critical.json, written by label_temporal_critical.py from
the per-question hand review, not from a keyword rule.

Macro metrics degrade on a subset: splitting 308 questions into 88 shrinks every class, so the
minimum support is printed and macro is suppressed when it drops below 5.

Usage:  python3 pipeline/critical_subset_report.py [dataset ...]     # default: medbullets mmlu
        RESULTS_DIR=results/revised_prompt python3 pipeline/critical_subset_report.py medbullets
"""
import json, os, statistics, sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))
from metrics import compute_metrics, aggregate, _labels_from_options

MODEL = os.environ.get("MODEL", "gpt-5.4-mini")
RES   = os.environ.get("RESULTS_DIR", "results/old_prompt")
METHODS = ["walker", "walker_interval", "raw_1hop", "raw_2hop", "tog", "hykge", "cot", "vanilla"]
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


def run_metrics(runs, keep, labels):
    """Per-run metrics restricted to `keep` uids, plus per-question mean accuracy over runs."""
    rm, perq = [], {}
    for r in runs:
        sel = [x for x in r["results"] if x["uid"] in keep]
        rm.append(compute_metrics(sel, labels=labels))
        for x in sel:
            perq.setdefault(x["uid"], []).append(1.0 if x["is_correct"] else 0.0)
    return rm, {u: statistics.fmean(v) for u, v in perq.items()}


for ds in (sys.argv[1:] or ["medbullets", "mmlu"]):
    tc = PIPE / f"datasets/{ds}/temporal_critical.json"
    bp = PIPE / f"datasets/{ds}/benchmark.json"
    if not tc.exists() or not bp.exists():
        print(f"\n### {ds}: 無人工判讀標記"); continue
    crit = set(json.load(open(tc))["uids"])
    bench = json.load(open(bp))
    labels = _labels_from_options(bench)
    allu = {b["uid"] for b in bench}

    rows = []
    for m in METHODS:
        f = PIPE / RES / f"{ds}_{m}_{MODEL.replace('/','_')}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        rm_all, pq_all = run_metrics(d["runs"], allu, labels)
        rm_cr,  pq_cr  = run_metrics(d["runs"], crit, labels)
        rows.append((m, aggregate(rm_all), aggregate(rm_cr), pq_all, pq_cr, rm_cr))
    if not rows:
        print(f"\n### {ds}: {RES} 內無結果"); continue

    base = next((r for r in rows if r[0] == "vanilla"), None)
    base_lift = (base[2]["accuracy"] - base[1]["accuracy"]) if base else None
    rows.sort(key=lambda r: -r[2]["accuracy"])
    n_crit = len(crit & allu)
    min_sup = rows[0][5][0]["min_support"]

    print(f"\n{'='*100}")
    print(f"### {ds} · temporal-critical 子集 · {RES} · {n_crit}/{len(allu)} 題"
          f"（{100*n_crit/len(allu):.1f}%）")
    print("=" * 100)

    hdr = (f"{'method':17}{'ALL %':>9}{'critical %':>13}{'lift':>8}"
           + (f"{'vs van':>9}" if base_lift is not None else "")
           + f"{'critMacroP':>12}{'critMacroR':>12}{'critParseP':>12}")
    print(f"\n{hdr}"); print("-" * len(hdr))
    for m, a_all, a_cr, _, _, _ in rows:
        lift = a_cr["accuracy"] - a_all["accuracy"]
        line = (f"{m:17}{a_all['accuracy']:>9.2f}{a_cr['accuracy']:>13.2f}{lift:>+8.2f}")
        if base_lift is not None:
            line += f"{lift - base_lift:>+9.2f}"
        line += f"{a_cr['macro_precision']:>12.2f}{a_cr['macro_recall']:>12.2f}{a_cr['parseable_precision']:>12.2f}"
        print(line)

    if base_lift is not None:
        print(f"\n  vanilla 在 critical 子集的 lift = {base_lift:+.2f}pp（完全沒有 KG）。")
        print(f"  這 {n_crit} 題本身就比較好答，所以「vs van」才是 KG 的貢獻 —— 正值才算證據。")
    else:
        print(f"\n  ⚠ 無 vanilla 結果，無法判斷 lift 是方法效果還是子集本身較易。")
    if min_sup < 5:
        print(f"  ⚠ 子集切完後最小類別只剩 {min_sup} 題 — critMacroP/R 不可靠，看 critParseP。")

    # paired test on the subset,每個方法 vs vanilla
    if base and wilcoxon:
        print(f"\n  critical 子集上，各方法 vs vanilla（逐題配對，跨 run 平均）")
        print(f"  {'method':17}{'Δ pp':>9}{'win':>6}{'loss':>6}{'tie':>6}{'p':>10}")
        print("  " + "-" * 54)
        for m, _, _, _, pq_cr, _ in rows:
            if m == "vanilla":
                continue
            common = sorted(set(pq_cr) & set(base[4]))
            a = [pq_cr[u] for u in common]; b = [base[4][u] for u in common]
            p = "—"
            if any(x != y for x, y in zip(a, b)):
                try: p = f"{wilcoxon(a, b).pvalue:.4f}"
                except Exception: pass
            win = sum(1 for x, y in zip(a, b) if x > y); loss = sum(1 for x, y in zip(a, b) if x < y)
            print(f"  {m:17}{100*(statistics.fmean(a)-statistics.fmean(b)):>+9.2f}"
                  f"{win:>6}{loss:>6}{len(common)-win-loss:>6}{p:>10}")
        print(f"  n={n_crit} — 看方向，不要當顯著性用。")
print()
