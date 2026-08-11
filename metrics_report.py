#!/usr/bin/env python3
"""Accuracy + precision / recall / F1 for stored reader results.

Definitions live in metrics.py. By default this scans results/old_prompt and
results/revised_prompt for MODEL=gpt-5.4-mini. For vLLM/block runs, set RESULTS_DIR to the
folder that contains the JSON files.

Set REPARSE=1 to ignore stored `predicted` values and re-extract answers from `raw_response` with
the current answer_extract.py. This is useful after parser fixes.

Usage:
  python3 metrics_report.py 329 medbullets
  RESULTS_DIR=results/vllm python3 metrics_report.py 329 1273
  REPARSE=1 RESULTS_DIR=results/old_prompt python3 metrics_report.py 329
  MODEL=gpt-5.4-mini DETAIL=1 python3 metrics_report.py 329
"""
import json, glob, os, sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))
from answer_extract import extract_letter
from metrics import compute_metrics, aggregate, _labels_from_options

MODEL  = os.environ.get("MODEL", "gpt-5.4-mini")
DETAIL = os.environ.get("DETAIL")
RESULTS_DIR = os.environ.get("RESULTS_DIR")
REPARSE = os.environ.get("REPARSE", "").lower() in ("1", "true", "yes", "on")
DATASETS = sys.argv[1:] or ["329", "1273", "medbullets", "mmlu"]


def _runs(d):
    if "runs" in d:
        return d["runs"]
    if "results" in d:
        return [{"run": d.get("run", 1), "results": d["results"]}]
    return []


def _result_sets(ds):
    if RESULTS_DIR:
        rd = PIPE / RESULTS_DIR
        return [(rd.name, sorted(glob.glob(str(rd / f"{ds}_*.json"))))]
    tag = MODEL.replace("/", "_")
    return [
        ("old_prompt", sorted(glob.glob(str(PIPE / "results" / "old_prompt" / f"{ds}_*_{tag}.json")))),
        ("revised_prompt", sorted(glob.glob(str(PIPE / "results" / "revised_prompt" / f"{ds}_*_{tag}.json")))),
    ]


def _reparse_rows(rows, options_by_uid):
    out = []
    for r in rows:
        rr = dict(r)
        rr["predicted"] = extract_letter(rr.get("raw_response") or "", options_by_uid.get(rr.get("uid"), {}))
        rr["is_correct"] = rr["predicted"] == rr.get("gold")
        out.append(rr)
    return out


for ds in DATASETS:
    bpath = PIPE / f"datasets/{ds}/benchmark.json"
    if not bpath.exists():
        continue
    bench = json.load(open(bpath))
    labels = _labels_from_options(bench)
    options_by_uid = {b["uid"]: b.get("options", {}) for b in bench}
    gold_dist = {}
    for b in bench:
        gold_dist[b["answer"]] = gold_dist.get(b["answer"], 0) + 1

    for sub, files in _result_sets(ds):
        rows = []
        for f in files:
            try:
                d = json.load(open(f))
            except Exception:
                continue
            runs = _runs(d)
            if not runs:
                continue
            scored_runs = []
            for r in runs:
                rr = r["results"]
                scored_runs.append(_reparse_rows(rr, options_by_uid) if REPARSE else rr)
            rm = [compute_metrics(rr, labels=labels) for rr in scored_runs]
            rows.append((d.get("method", Path(f).stem).replace("__revised", ""), aggregate(rm), rm))
        if not rows:
            continue
        rows.sort(key=lambda t: -t[1]["accuracy"])
        a0 = rows[0][1]

        print(f"\n{'='*94}")
        print(f"### {ds} · {sub}" + (" · reparsed" if REPARSE else "") +
              f"   n={a0['n']}  runs={a0['n_runs']}  classes={a0['n_classes']} "
              f"(min support {a0['min_support']})")
        print("=" * 94)
        hdr = (f"{'method':25}{'Accuracy':>16}{'MacroP':>16}{'MacroR':>16}"
               f"{'ParseableP':>16}{'MacroF1':>10}{'WeightF1':>10}{'unparse':>12}")
        print(hdr); print("-" * len(hdr))
        for m, a, _ in rows:
            cell = lambda k: f"{a[k]:.2f} +/- {a[k+'_std']:.2f}"
            print(f"{m:25}{cell('accuracy'):>16}{cell('macro_precision'):>16}"
                  f"{cell('macro_recall'):>16}{cell('parseable_precision'):>16}"
                  f"{a['macro_f1']:>10.2f}{a['weighted_f1']:>10.2f}"
                  f"{str(a['unparseable']):>12}")
        if a0["min_support"] < 5:
            print(f"\n  note: min class support is {a0['min_support']}; macro_* is unstable. Gold dist: {gold_dist}")

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
