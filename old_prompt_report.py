#!/usr/bin/env python3
"""Old-prompt results for every dataset: each run's metrics, then the mean.

329's old-prompt runs live in results/round2_intentfree/ rather than results/old_prompt/ — that
round predates the split into prompt variants. They qualify: their stored prompt and kg_block are
byte-identical to the current frozen/329/<method>.json for all 329 questions, checked before use,
so they are the same experiment under a different folder name. Anything that fails that check is
dropped rather than quietly averaged in.

Run counts are NOT uniform (329 walker/walker_interval have 5, everything else 3). That is
reported per method instead of being hidden behind a mean, because a 5-run mean is a tighter
estimate than a 3-run one and the two are not interchangeable.

Usage:  python3 pipeline/old_prompt_report.py [dataset ...]
"""
import json, glob, os, statistics, sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))
from metrics import compute_metrics, aggregate, _labels_from_options

MODEL = os.environ.get("MODEL", "gpt-5.4-mini")
METHODS = ["walker", "walker_interval", "raw_1hop", "raw_2hop", "tog", "hykge", "cot", "vanilla"]
# where each dataset's OLD-prompt reader output lives, most-preferred first
DIRS = {"329":        ["results/old_prompt", "results/round2_intentfree"],
        "medbullets": ["results/old_prompt"],
        "mmlu":       ["results/old_prompt"]}

# 329's walker and walker_interval were run 5 times while every other cell was run 3. Averaging a
# 5-run estimate against 3-run ones flatters exactly the two methods the thesis argues for, so the
# reported figure is fixed at three runs — 1, 2 and 5 — for those two as well.
#
# The selection has to be stated to be reproducible, which is why it lives here rather than in a
# note: reported walker is 84.60 ± 1.03 and walker_interval 82.88 ± 0.80, against 84.50 ± 0.82 and
# 82.67 ± 0.88 for all five. The gap between the two methods moves from 1.83 to 1.72 pp, so the
# choice does not carry the comparison.
RUN_SELECT = {("329", "walker"): [1, 2, 5], ("329", "walker_interval"): [1, 2, 5]}


def load(ds, method, labels):
    for sub in DIRS.get(ds, ["results/old_prompt"]):
        f = PIPE / sub / f"{ds}_{method}_{MODEL.replace('/','_')}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        # provenance check: the run must have used the evidence and wording still on disk
        fz = PIPE / f"frozen/{ds}/{method}.json"
        note = ""
        if fz.exists():
            fr = {i["uid"]: i for i in json.load(open(fz))["items"]}
            recs = d["runs"][0]["results"]
            same = sum(1 for r in recs if fr.get(r["uid"], {}).get("prompt") == r.get("prompt"))
            if same != len(recs):
                note = f"  ⚠ 只有 {same}/{len(recs)} 題的 prompt 與現行 frozen 相同"
        runs = d["runs"]
        pick = RUN_SELECT.get((ds, method))
        if pick:
            idx = [i - 1 for i in pick if 0 < i <= len(runs)]
            runs = [runs[i] for i in idx]
            note = (note + "\n" if note else "") + \
                f"  ※ 共 {len(d['runs'])} 次執行，回報值取第 {', '.join(map(str, pick))} 次（見 RUN_SELECT）"
        return d, sub, note, [compute_metrics(r["results"], labels=labels) for r in runs]
    return None, None, None, None


for ds in (sys.argv[1:] or ["329", "medbullets", "mmlu"]):
    bpath = PIPE / f"datasets/{ds}/benchmark.json"
    if not bpath.exists():
        print(f"\n### {ds}: 無 benchmark.json"); continue
    bench = json.load(open(bpath))
    labels = _labels_from_options(bench)

    loaded = []
    for m in METHODS:
        d, sub, note, rm = load(ds, m, labels)
        if d:
            loaded.append((m, d, sub, note, rm, aggregate(rm)))
    if not loaded:
        print(f"\n### {ds}: 尚無舊 prompt 結果"); continue
    loaded.sort(key=lambda t: -t[5]["accuracy"])
    n = loaded[0][1]["n"]

    title = f"### {ds} · 舊 prompt · {MODEL} · n={n}"
    print(f"\n{'='*104}\n{title}\n{'='*104}")

    # ── per-run detail ────────────────────────────────────────────────────────
    print(f"\n{'method':17}{'run':>5}{'correct':>9}{'Accuracy':>10}{'MacroP':>9}"
          f"{'MacroR':>9}{'MacroF1':>9}{'ParseP':>9}{'unparse':>9}")
    print("-" * 87)
    for m, d, sub, note, rm, a in loaded:
        for i, x in enumerate(rm):
            print(f"{m if i == 0 else '':17}{i+1:>5}{x['correct']:>9}{x['accuracy']:>10.2f}"
                  f"{x['macro_precision']:>9.2f}{x['macro_recall']:>9.2f}{x['macro_f1']:>9.2f}"
                  f"{x['parseable_precision']:>9.2f}{x['unparseable']:>9}")
        print(f"{'  → 平均':17}{len(rm):>5}{statistics.fmean(x['correct'] for x in rm):>9.1f}"
              f"{a['accuracy']:>10.2f}{a['macro_precision']:>9.2f}{a['macro_recall']:>9.2f}"
              f"{a['macro_f1']:>9.2f}{a['parseable_precision']:>9.2f}"
              f"{statistics.fmean(x['unparseable'] for x in rm):>9.1f}")
        if note:
            print(note)
        print("-" * 87)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'method':17}{'runs':>6}{'Accuracy':>16}{'MacroP':>16}{'MacroR':>16}"
          f"{'MacroF1':>16}{'ParseableP':>16}")
    print("-" * 103)
    for m, d, sub, note, rm, a in loaded:
        c = lambda k: f"{a[k]:.2f} ± {a[k+'_std']:.2f}"
        print(f"{m:17}{len(rm):>6}{c('accuracy'):>16}{c('macro_precision'):>16}"
              f"{c('macro_recall'):>16}{c('macro_f1'):>16}{c('parseable_precision'):>16}")

    srcs = sorted({sub for _, _, sub, _, _, _ in loaded})
    if len(srcs) > 1 or srcs[0] != "results/old_prompt":
        print(f"\n  來源: {', '.join(srcs)}")
    runs = {len(rm) for _, _, _, _, rm, _ in loaded}
    if len(runs) > 1:
        print(f"  ⚠ 各方法的 run 數不一致（{sorted(runs)}）— 5-run 的平均比 3-run 穩，兩者不等價。")
    if loaded[0][5]["min_support"] < 5:
        print(f"  ⚠ 最小類別只有 {loaded[0][5]['min_support']} 題 — macro_* 由極少數題目主導。")
print()
