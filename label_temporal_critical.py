#!/usr/bin/env python3
"""Write the temporal-critical labels from the hand review into the dataset itself.

The judgement currently lives only in verification/manual_read_<ds>.jsonl, which means every
downstream consumer has to know that file exists and how to parse it. Putting the label on the
question instead makes it travel with the data: anyone who loads benchmark.json gets it, and a
subset can be taken without a join.

Writes two things per dataset:
  datasets/<ds>/benchmark.json          + "temporal_critical" (bool) and "temporal_axis" (str|null)
  datasets/<ds>/temporal_critical.json  the uid list on its own, for quick filtering

benchmark.json is rewritten in place, so the existing keys are preserved and only these two are
added — the retrieval pipeline reads uid/question/options/answer and ignores extras.

Usage:  python3 pipeline/label_temporal_critical.py [dataset ...]   # default: medbullets mmlu
"""
import json, sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
DATASETS = sys.argv[1:] or ["medbullets", "mmlu"]

for ds in DATASETS:
    hand = PIPE / f"verification/manual_read_{ds}.jsonl"
    if not hand.exists():
        print(f"[{ds}] skip — no hand review at {hand}")
        continue
    review = {}
    for line in open(hand):
        r = json.loads(line)
        review[r["uid"]] = r

    bpath = PIPE / f"datasets/{ds}/benchmark.json"
    bench = json.load(open(bpath))
    missing = [b["uid"] for b in bench if b["uid"] not in review]
    if missing:
        # a partial labelling would silently mark unreviewed questions as not-critical, which is a
        # different claim from "not yet looked at" — refuse rather than guess
        print(f"[{ds}] ABORT — {len(missing)} 題沒有人工判讀（例: {missing[:3]}）")
        continue

    n_crit = 0
    for b in bench:
        r = review[b["uid"]]
        b["temporal_critical"] = (r["verdict"] == "critical")
        b["temporal_axis"] = r.get("axis") if r["verdict"] == "critical" else None
        n_crit += b["temporal_critical"]
    json.dump(bench, open(bpath, "w"), indent=1)

    uids = [b["uid"] for b in bench if b["temporal_critical"]]
    json.dump({"dataset": ds, "n_total": len(bench), "n_critical": len(uids),
               "source": f"verification/manual_read_{ds}.jsonl (逐題人工判讀)",
               "uids": uids},
              open(PIPE / f"datasets/{ds}/temporal_critical.json", "w"), indent=1)
    print(f"[{ds}] {n_crit}/{len(bench)} critical → benchmark.json (+2 欄) 與 temporal_critical.json")
