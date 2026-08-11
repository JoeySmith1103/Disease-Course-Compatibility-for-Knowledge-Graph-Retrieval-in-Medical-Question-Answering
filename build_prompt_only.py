#!/usr/bin/env python3
"""Freeze the prompt-only methods (no retrieval, no Neo4j): vanilla, cot.

These just wrap each question in prompts.VANILLA / prompts.COT_MINIMAL — there is nothing to
retrieve, so build_kg / freeze_baseline don't apply. This closes the last frozen-file gap so the
whole set is reproducible from pipeline/ alone.

Usage:  DATASET=1273|329  python3 pipeline/build_prompt_only.py
Output: frozen/<ds>/vanilla.json, frozen/<ds>/cot.json  ({uid, gold, route, kg_block:"", prompt})
"""
import json, os, importlib.util
from pathlib import Path

PIPE = Path(__file__).resolve().parent
DS = os.environ.get("DATASET", "1273")
DD = PIPE / f"datasets/{DS}"

_spec = importlib.util.spec_from_file_location("P", PIPE / "prompts.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)

bench = json.load(open(DD / "benchmark.json"))
def ob(o): return "\n".join(f"  {k}. {v}" for k, v in sorted(o.items()))

for method, template in (("vanilla", P.VANILLA), ("cot", P.COT_MINIMAL)):
    items = [{"uid": b["uid"], "gold": b["answer"], "route": method, "kg_block": "",
              "prompt": template.format(question=b["question"], options_block=ob(b["options"]))}
             for b in bench]
    out = PIPE / f"frozen/{DS}/{method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"method": method, "dataset": DS, "n": len(items), "items": items},
              open(out, "w"), indent=1)
    print(f"froze {len(items)} -> {out}")
