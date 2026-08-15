#!/usr/bin/env python3
"""One table over the five axes a single utility has to win on at once.

329 is not sliced. The whole benchmark was built as the temporal-critical set, so a "critical
subset" of it would be a subset of a subset — the dataset IS the axis. MedBullets and MMLU are
general benchmarks, so each contributes two axes: the full set, and the hand-labelled
temporal-critical questions inside it.

    329 (329)   MedBullets (308)   MedBullets-crit (88)   MMLU (272)   MMLU-crit (51)

The bar on each axis is the best EXTERNAL baseline — vanilla / cot / tog / hykge / raw_1hop /
raw_2hop / medrag. The walker family is the method under test, so beating the shipped walker is a
separate (weaker) question and is not what "beats all baselines" means.

Baselines resolve exactly as old_prompt_report.py resolves them, including the MedBullets raw_1hop
override to the middle of its three same-condition N=3 runs, so the numbers here and there agree.

Variants are shown per BATCH, never averaged across batches: the same frozen prompt has moved
2.81pp between batches, so a cross-batch mean is a mean over dates, not over methods. Where a
variant appears in several batches, each is a separate row and the disagreement between them is
the honest error bar.

Usage:  python3 pipeline/five_axis_report.py            # variants with N>=3
        MIN_RUNS=1 python3 pipeline/five_axis_report.py # include N=1 screens
"""
import glob, json, os, statistics, sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
MODEL = os.environ.get("MODEL", "gpt-5.4-mini")
MIN_RUNS = int(os.environ.get("MIN_RUNS", "3"))
BASELINES = ["raw_1hop", "raw_2hop", "medrag", "cot", "tog", "hykge", "vanilla"]

# same resolution as old_prompt_report.py — see FILE_ALIAS there for why each override exists
ALIAS = {("medbullets", "raw_1hop"): ("results/param_sweep_n3", "raw_1hop"),
         ("medbullets", "medrag"):   ("results/judged", "medrag")}
DIRS = {"329": ["results/old_prompt", "results/round2_intentfree"],
        "medbullets": ["results/old_prompt"], "mmlu": ["results/old_prompt"]}

# (label, dataset, uid filter) — 329 contributes one axis, the other two contribute two each
def _crit(ds):
    return set(json.load(open(PIPE / f"datasets/{ds}/temporal_critical.json"))["uids"])

AXES = [("329", "329", None),
        ("MedBullets", "medbullets", None),
        ("MB-crit", "medbullets", _crit("medbullets")),
        ("MMLU", "mmlu", None),
        ("MMLU-crit", "mmlu", _crit("mmlu"))]


def acc(path, keep):
    """Mean accuracy over runs, restricted to `keep` uids (None = all)."""
    d = json.load(open(path))
    if not isinstance(d, dict) or "runs" not in d:
        return None, 0
    per = []
    for r in d["runs"]:
        sel = [x for x in r["results"] if keep is None or x["uid"] in keep]
        if not sel:
            return None, 0
        per.append(100 * sum(x["is_correct"] for x in sel) / len(sel))
    return statistics.fmean(per), len(d["runs"])


def find(ds, method):
    key = ALIAS.get((ds, method))
    dirs = [key[0]] if key else DIRS[ds]
    method = key[1] if key else method
    for d in dirs:
        p = PIPE / d / f"{ds}_{method}_{MODEL.replace('/','_')}.json"
        if p.exists():
            return p
    return None


# ── bar: the best external baseline on each axis ─────────────────────────────
bar, bar_who = {}, {}
base_rows = {}
for b in BASELINES:
    row = {}
    for label, ds, keep in AXES:
        p = find(ds, b)
        row[label] = acc(p, keep)[0] if p else None
    base_rows[b] = row
for label, _, _ in AXES:
    vals = [(row[label], b) for b, row in base_rows.items() if row.get(label) is not None]
    bar[label], bar_who[label] = max(vals)

W = 44
head = f"{'method':{W}s}" + "".join(f"{l:>13s}" for l, _, _ in AXES)
print("五個軸上的同時比較（329 本身就是 temporal-critical 集，不再切子集）\n")
print(head)
print("-" * len(head))
for b in BASELINES:
    cells = "".join(f"{base_rows[b][l]:13.2f}" if base_rows[b].get(l) is not None else f"{'—':>13s}"
                    for l, _, _ in AXES)
    print(f"{('baseline  ' + b)[:W-1]:{W}s}{cells}")
print("-" * len(head))
print(f"{'門檻（最高外部 baseline）':{W-8}s}" + "".join(f"{bar[l]:13.2f}" for l, _, _ in AXES))
print(f"{'':{W}s}" + "".join(f"{bar_who[l][:12]:>13s}" for l, _, _ in AXES))
print()

# ── variants, one row per batch ──────────────────────────────────────────────
seen = {}
for p in glob.glob(str(PIPE / "results/**/*_gpt-5.4-mini.json"), recursive=True):
    if "_archive" in p or "_staging" in p or "revised" in p:
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if not isinstance(d, dict) or "runs" not in d:
        continue
    ds, m = d.get("dataset"), d.get("method")
    if ds not in DIRS or m in BASELINES or len(d["runs"]) < MIN_RUNS:
        continue
    batch = os.path.dirname(p).replace(str(PIPE) + "/results/", "").replace(str(PIPE) + "/results", ".")
    seen.setdefault((m, batch), {})
    for label, dds, keep in AXES:
        if dds != ds:
            continue
        a, n = acc(p, keep)
        if a is not None:
            seen[(m, batch)][label] = (a, n)

rows = []
for (m, batch), cells in seen.items():
    passed = sum(1 for l, _, _ in AXES if l in cells and cells[l][0] > bar[l])
    tested = sum(1 for l, _, _ in AXES if l in cells)
    rows.append((passed, tested, m, batch, cells))

print(f"變體（每個批次一列，不跨批平均；✔ = 超過該軸門檻）    N>={MIN_RUNS}\n")
print(head + "   過/測")
print("-" * (len(head) + 8))
for passed, tested, m, batch, cells in sorted(rows, key=lambda r: (-r[0], -r[1], -sum(
        c[0] for c in r[4].values()) / max(len(r[4]), 1))):
    if tested < 2:
        continue
    out = ""
    for l, _, _ in AXES:
        if l not in cells:
            out += f"{'—':>13s}"
        else:
            a, n = cells[l]
            out += f"{a:11.2f}{'✔' if a > bar[l] else ' '} "
    print(f"{(m + '  @' + batch)[:W-1]:{W}s}{out}   {passed}/{tested}")
