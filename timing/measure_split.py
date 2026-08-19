#!/usr/bin/env python3
"""Split each method's per-question cost into retrieval and answering, without touching the drivers.

Every driver runs retrieval and answering inside one process_one(), and none has a retrieval-only
switch. Rather than edit code that produced the published numbers, this wraps llm_client.call_llm,
records the wall time of each call, and attributes the LAST call of each question to answering and
everything before it to retrieval. That split is exact for these drivers: all of them answer once,
at the end.

The per-disease duration LLM calls are excluded -- they are cached, shared across questions and
datasets, and are a one-time build rather than part of answering a question. BC_CACHE_ONLY forces a
miss to score 0 instead of generating.

Usage:  python3 pipeline/timing/measure_split.py tog hykge medrag
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(PIPE)); sys.path.insert(0, str(PIPE / "code"))
sys.path.insert(0, str(PIPE / "code" / "dkr_policy"))
SAMPLE = HERE / "bench_329_sample25.json"
DD = PIPE / "datasets/329"
os.environ.update(BENCH_PATH=str(SAMPLE), PD_PATH=str(DD/"durations.json"),
                  SEEDS_PATH=str(DD/"seeds.json"), SYM_PATH=str(DD/"symptoms.json"),
                  QE_PATH=str(DD/"query_entities.json"), MEDRAG_PD=str(DD/"durations.json"),
                  TOG_OUT_TAG="timing_tog", HYKGE_OUT_TAG="timing_hykge",
                  MEDRAG_OUT_TAG="timing_medrag",
                  DKR_MATRIX_PKL=str(PIPE/"cache/umls_broad_embeddings_sapbert.pkl"),
                  BC_CACHE_ONLY="1", WALKER_NO_ONDEMAND="1")

import llm_client
_orig = llm_client.call_llm
CALLS = []          # (t_seconds,) appended in order for the question being processed


def _timed(*a, **k):
    t0 = time.perf_counter()
    try:
        return _orig(*a, **k)
    finally:
        CALLS.append(time.perf_counter() - t0)


llm_client.call_llm = _timed
try:
    import spectrum_textbook
    spectrum_textbook.call_llm = _timed          # hykge imports it from here
except Exception:
    pass

bench = json.load(open(SAMPLE))


def run(name, mod_name, attr="process_one"):
    mod = __import__(mod_name)
    fn = getattr(mod, attr)
    rows = []
    for it in bench:
        CALLS.clear()
        t0 = time.perf_counter()
        try:
            fn(it)
        except Exception as e:
            print(f"    {it['uid']}: {type(e).__name__} {e}"); continue
        total = time.perf_counter() - t0
        answer = CALLS[-1] if CALLS else 0.0
        llm_in_loop = sum(CALLS[:-1]) if len(CALLS) > 1 else 0.0
        rows.append(dict(uid=it["uid"], total=total, answer=answer,
                         retrieval=total - answer, llm_calls=len(CALLS),
                         llm_in_loop=llm_in_loop))
    if not rows:
        print(f"{name}: 全部失敗"); return
    import statistics as st
    med = lambda k: st.median(r[k] for r in rows)
    out = dict(method=name, n=len(rows),
               retrieval_s=round(med("retrieval"), 3), answer_s=round(med("answer"), 3),
               total_s=round(med("total"), 3), llm_calls_median=med("llm_calls"),
               llm_in_loop_s=round(med("llm_in_loop"), 3), rows=rows)
    json.dump(out, open(HERE / f"split_{name}.json", "w"), indent=1)
    print(f"{name:8s} 檢索 {out['retrieval_s']:6.2f}s  作答 {out['answer_s']:5.2f}s  "
          f"合計 {out['total_s']:6.2f}s  檢索內 LLM 呼叫 {out['llm_calls_median']-1:.0f} 次"
          f"（{out['llm_in_loop_s']:.2f}s）  n={out['n']}")


TARGETS = {"medrag": "run_medrag_textbook", "tog": "tog_baseline", "hykge": "hykge_baseline"}
for name in (sys.argv[1:] or list(TARGETS)):
    run(name, TARGETS[name])
