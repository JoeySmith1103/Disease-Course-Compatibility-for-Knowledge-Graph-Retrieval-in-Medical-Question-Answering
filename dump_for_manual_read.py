#!/usr/bin/env python3
"""Print questions in compact form so they can be read and judged one by one BY HAND.

This exists because the scripted verification (verify_duration_critical.py) delegates the judgement
to gpt-5.4-mini. That measures what a model does with the question; it is not the same as someone
having read the question. This dump is for the second thing.

Usage:  DATASET=medbullets START=0 N=25 python3 pipeline/dump_for_manual_read.py
"""
import json, os, textwrap
from pathlib import Path

PIPE = Path(__file__).resolve().parent
DS    = os.environ.get("DATASET", "medbullets")
START = int(os.environ.get("START", "0"))
N     = int(os.environ.get("N", "25"))

bench = json.load(open(PIPE / f"datasets/{DS}/benchmark.json"))
for b in bench[START:START + N]:
    print(f"\n===== {b['uid']}  [gold={b['answer']}] " + "=" * 40)
    print(textwrap.fill(b["question"], 112))
    for k, v in sorted(b["options"].items()):
        mark = "*" if k == b["answer"] else " "
        print(f"  {mark}{k}. {v}")
print(f"\n[{DS}] shown {START}..{min(START+N, len(bench))} of {len(bench)}")
