#!/usr/bin/env python3
"""Time the per-question LLM preprocessing this method needs before retrieval can start.

HyKGE generates one hypothesis per question inside its retrieval loop. This method needs the same
kind of thing, but four of them, produced ahead of time: seeds (the differential-diagnosis
hypotheses that become walk start points), symptoms (the SapBERT query), query_entities (the wider
entity set the full-query cos is measured against), and durations (the patient's elapsed time, the
input to bc). Leaving them out of the timing would compare our retrieval against HyKGE's
retrieval-plus-hypothesis, which is not the same quantity.

Both totals are reported because they answer different questions:
  sequential   what one question costs if the four are issued one after another
  parallel     max of the four — they share no inputs, so a deployment would issue them together

Excluded, as everywhere else in this timing set: the per-disease duration generation behind bc.
That cache is built once and shared across every question and dataset.

Usage:  python3 pipeline/timing/measure_preprocess.py [n_questions]
"""
import json, os, sys, time, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(PIPE)); sys.path.insert(0, str(PIPE / "code"))
from llm_client import call_llm

import importlib.util as _iu
_s = _iu.spec_from_file_location("prep", PIPE / "prepare_dataset_inputs.py")
_m = _iu.module_from_spec(_s)
_m.__dict__["__name__"] = "prep"
try:
    _s.loader.exec_module(_m)
except SystemExit:
    pass
PROMPTS = {"seeds": _m.SEEDS_PROMPT, "symptoms": _m.SYMPTOMS_PROMPT, "query_entities": _m.QE_PROMPT}

sys.path.insert(0, str(PIPE / "code"))
import extract_patient_duration_llm as EPD
PROMPTS["durations"] = None            # built per question by EPD._build_extraction_prompt

MODEL = os.environ.get("MODEL", "gpt-5.4-mini")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
bench = json.load(open(HERE / "bench_329_sample25.json"))[:N]

per_call = {k: [] for k in PROMPTS}
rows = []
for it in bench:
    q = it["question"]
    times = {}
    for k, tmpl in PROMPTS.items():
        p = EPD._build_extraction_prompt(q) if k == "durations" else tmpl.format(question=q[:3000])
        t0 = time.perf_counter()
        try:
            call_llm(p, model=MODEL)
        except Exception as e:
            print(f"  {it['uid']} {k}: {type(e).__name__}")
        dt = time.perf_counter() - t0
        times[k] = dt; per_call[k].append(dt)
    rows.append(dict(uid=it["uid"], **{k: round(v, 3) for k, v in times.items()},
                     sequential=round(sum(times.values()), 3), parallel=round(max(times.values()), 3)))

out = dict(n=len(rows), model=MODEL,
           per_call_median={k: round(st.median(v), 3) for k, v in per_call.items()},
           sequential_median=round(st.median(r["sequential"] for r in rows), 3),
           parallel_median=round(st.median(r["parallel"] for r in rows), 3),
           excluded="per-disease duration generation behind bc (cached, shared)", rows=rows)
json.dump(out, open(HERE / "preprocess.json", "w"), indent=1)
print("每次呼叫中位數:", out["per_call_median"])
print(f"四次序列 {out['sequential_median']:.2f}s   四次平行(取 max) {out['parallel_median']:.2f}s   n={out['n']}")
