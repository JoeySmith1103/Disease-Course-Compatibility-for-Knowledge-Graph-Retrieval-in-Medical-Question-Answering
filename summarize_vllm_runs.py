#!/usr/bin/env python3
"""Summarize vLLM batch result files into CSV and Markdown tables.

This matches the filename convention used by run_reader_vllm_batch.py. It is a local helper and is
ignored by git in this clone.

Example:
  DATASETS=329,medbullets,mmlu MODEL=Qwen/Qwen3.5-9B RUNS=1,2,3 \
  RESULTS_DIR=results/vllm_qwen_block OUT_PREFIX=qwen_block_summary \
  python3 summarize_vllm_runs.py
"""
import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))
from metrics import compute_metrics, _labels_from_options

DATASETS = [x for x in os.environ.get("DATASETS", "329,medbullets,mmlu").split(",") if x]
RUNS = [x for x in os.environ.get("RUNS", "1,2,3").split(",") if x]
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
PROMPT_VARIANT = os.environ.get("PROMPT_VARIANT")
if not PROMPT_VARIANT:
    PROMPT_VARIANT = (os.environ.get("PROMPT_VARIANTS") or "small_model_block").split(",")[0]
RESULTS_DIR = PIPE / os.environ.get("RESULTS_DIR", "results")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "vllm_summary")

DEFAULT_METHOD_FILES = {
    "vanilla": "vanilla",
    "cot": "cot",
    "medrag": "medrag",
    "raw_1hop": "raw_1hop__revised",
    "raw_2hop": "raw_2hop__revised",
    "tog": "tog__revised",
    "hykge": "hykge__revised",
    "walker": "walker__revised",
    "walker_interval": "walker_interval__revised",
}
METHODS = [x for x in os.environ.get(
    "METHODS",
    "vanilla,cot,raw_1hop,raw_2hop,medrag,tog,hykge,walker,walker_interval",
).split(",") if x]


def tag(text):
    base = Path(str(text).rstrip("/")).name or str(text)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base.replace(":", "_"))


def method_file(method):
    return DEFAULT_METHOD_FILES.get(method, method)


def result_path(ds, mfile, run):
    prompt_part = "" if PROMPT_VARIANT == "small_model_block" else f"_{tag(PROMPT_VARIANT)}"
    return RESULTS_DIR / f"{ds}_{mfile}_{tag(MODEL)}{prompt_part}_vllm_batch_run{run}.json"


def fmt(x):
    return "" if x is None else f"{x:.2f}"


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0


def labels_for(ds):
    bench = json.load(open(PIPE / f"datasets/{ds}/benchmark.json"))
    return _labels_from_options(bench)


def records_from(data):
    if data.get("results") is not None:
        return data["results"]
    runs = data.get("runs") or []
    return runs[0].get("results", []) if runs else []


def main():
    rows = []
    for ds in DATASETS:
        bpath = PIPE / f"datasets/{ds}/benchmark.json"
        if not bpath.exists():
            continue
        labels = labels_for(ds)
        for method in METHODS:
            mfile = method_file(method)
            run_metrics = []
            missing = []
            for run in RUNS:
                rp = result_path(ds, mfile, run)
                if not rp.exists():
                    missing.append(run)
                    run_metrics.append(None)
                    continue
                data = json.load(open(rp))
                run_metrics.append(compute_metrics(records_from(data), labels=labels))

            acc_m, acc_s = mean_std([m["accuracy"] if m else None for m in run_metrics])
            prec_m, prec_s = mean_std([m["macro_precision"] if m else None for m in run_metrics])
            rec_m, rec_s = mean_std([m["macro_recall"] if m else None for m in run_metrics])
            parse_m, parse_s = mean_std([m["parseable_precision"] if m else None for m in run_metrics])
            n = next((m["n"] for m in run_metrics if m), 0)
            status = "ok" if not missing else f"missing runs {','.join(missing)}"
            row = {
                "dataset": ds,
                "method": method,
                "method_file": mfile,
                "n": n,
                "status": status,
                "acc_mean": acc_m,
                "acc_std": acc_s,
                "macro_precision_mean": prec_m,
                "macro_precision_std": prec_s,
                "macro_recall_mean": rec_m,
                "macro_recall_std": rec_s,
                "parseable_precision_mean": parse_m,
                "parseable_precision_std": parse_s,
                "unparseable_runs": ",".join(str(m["unparseable"]) for m in run_metrics if m),
            }
            for i, run in enumerate(RUNS, 1):
                m = run_metrics[i - 1]
                row[f"run{i}_id"] = run
                row[f"run{i}_acc"] = m["accuracy"] if m else None
                row[f"run{i}_macro_precision"] = m["macro_precision"] if m else None
                row[f"run{i}_macro_recall"] = m["macro_recall"] if m else None
                row[f"run{i}_parseable_precision"] = m["parseable_precision"] if m else None
                row[f"run{i}_unparseable"] = m["unparseable"] if m else None
            rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{OUT_PREFIX}.csv"
    fieldnames = [
        "dataset", "method", "method_file", "n", "status",
        "run1_acc", "run1_macro_precision", "run1_macro_recall", "run1_parseable_precision", "run1_unparseable",
        "run2_acc", "run2_macro_precision", "run2_macro_recall", "run2_parseable_precision", "run2_unparseable",
        "run3_acc", "run3_macro_precision", "run3_macro_recall", "run3_parseable_precision", "run3_unparseable",
        "acc_mean", "acc_std", "macro_precision_mean", "macro_precision_std",
        "macro_recall_mean", "macro_recall_std", "parseable_precision_mean",
        "parseable_precision_std", "unparseable_runs",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) if isinstance(row.get(k), float) else row.get(k, "") for k in fieldnames})

    md_path = RESULTS_DIR / f"{OUT_PREFIX}.md"
    with open(md_path, "w") as f:
        f.write("# vLLM Summary\n\n")
        f.write("Precision and recall are macro metrics over answer letters.\n\n")
        f.write("| dataset | method | n | run1 acc/P/R | run2 acc/P/R | run3 acc/P/R | acc mean +/- std | P mean +/- std | R mean +/- std | unparseable | status |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            cells = []
            for i in range(1, 4):
                vals = [row.get(f"run{i}_acc"), row.get(f"run{i}_macro_precision"), row.get(f"run{i}_macro_recall")]
                cells.append("/".join(fmt(v) for v in vals))
            f.write(
                f"| {row['dataset']} | {row['method']} | {row['n']} | "
                f"{cells[0]} | {cells[1]} | {cells[2]} | "
                f"{fmt(row['acc_mean'])} +/- {fmt(row['acc_std'])} | "
                f"{fmt(row['macro_precision_mean'])} +/- {fmt(row['macro_precision_std'])} | "
                f"{fmt(row['macro_recall_mean'])} +/- {fmt(row['macro_recall_std'])} | "
                f"{row['unparseable_runs']} | {row['status']} |\n"
            )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
