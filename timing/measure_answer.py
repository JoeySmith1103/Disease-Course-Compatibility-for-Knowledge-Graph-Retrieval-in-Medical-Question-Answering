#!/usr/bin/env python3
"""Time the answering call for every method by replaying its frozen prompt.

The retrieval split (measure_split.py) already timed answering for the three methods whose drivers
it ran, but each of those went through its own code path. Replaying the frozen prompts puts every
method through one identical path, so the answer column compares like with like -- the only thing
that differs between rows is the prompt each method produced.

Usage:  python3 pipeline/timing/measure_answer.py [method ...]
"""
import json, os, sys, time, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(PIPE / "code"))
from llm_client import call_llm

MODEL = os.environ.get("MODEL", "gpt-5.4-mini")
UIDS = [x["uid"] for x in json.load(open(HERE / "bench_329_sample25.json"))]
METHODS = sys.argv[1:] or ["vanilla", "cot", "raw_1hop", "raw_2hop", "ours", "walker", "medrag", "hykge", "tog"]

out = {}
for m in METHODS:
    f = PIPE / f"frozen/329/{m}.json"
    if not f.exists():
        print(f"{m:10s} (無 frozen)"); continue
    pr = {i["uid"]: i.get("prompt") for i in json.load(open(f))["items"]}
    ts = []
    for u in UIDS:
        p = pr.get(u)
        if not p:
            continue
        t0 = time.perf_counter()
        try:
            call_llm(p, model=MODEL)
        except Exception as e:
            print(f"    {u}: {type(e).__name__}"); continue
        ts.append(time.perf_counter() - t0)
    if not ts:
        continue
    out[m] = dict(n=len(ts), median=round(st.median(ts), 3), mean=round(st.fmean(ts), 3),
                  p90=round(sorted(ts)[int(0.9 * len(ts))], 3))
    print(f"{m:10s} 作答中位數 {out[m]['median']:5.2f}s  平均 {out[m]['mean']:5.2f}s  p90 {out[m]['p90']:5.2f}s  n={out[m]['n']}")
json.dump(out, open(HERE / "answer.json", "w"), indent=1)
