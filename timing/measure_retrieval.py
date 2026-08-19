#!/usr/bin/env python3
"""Wall-clock retrieval cost per question for every method, on one fixed sample of MedQA 329.

The advisor's question is whether this method is cheap relative to the baselines, and nothing in
the pipeline recorded time -- every driver reported accuracy only. This runs each retrieval stage
over the same 25 questions and reports seconds per question.

WHAT IS EXCLUDED, deliberately: the LLM calls that generate a per-disease duration for the bc term.
They are cached and shared across every question and dataset, so charging them to a single run's
retrieval would misrepresent the marginal cost -- and the cache is a one-time build, not part of
answering a question. The environment forces cache-only so a miss scores 0 instead of silently
calling out and inflating the number.

WHAT IS INCLUDED: Neo4j traversal, SapBERT scoring, BM25, and -- for ToG and HyKGE -- the LLM calls
those methods make *inside* the retrieval loop, because that is intrinsic to how they retrieve
rather than an offline artefact.

Usage:  python3 pipeline/timing/measure_retrieval.py [method ...]
Output: pipeline/timing/retrieval_<method>.json
"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
SAMPLE = HERE / "bench_329_sample25.json"
DD = PIPE / "datasets/329"
N = len(json.load(open(SAMPLE)))

# DATASET matters as much as BENCH_PATH: build_raw_hops.py defaults to DATASET=1273 and writes to
# frozen/<DATASET>/, so setting only BENCH_PATH made it read the sample and overwrite 1273's frozen
# files. Both are pinned here, and every driver is given an output tag so nothing lands on a path a
# published number was read from.
BASE = dict(os.environ,
            DATASET="329",
            BENCH_PATH=str(SAMPLE),
            PD_PATH=str(DD / "durations.json"),
            SEEDS_PATH=str(DD / "seeds.json"),
            SYM_PATH=str(DD / "symptoms.json"),
            QE_PATH=str(DD / "query_entities.json"),
            DKR_MATRIX_PKL=str(PIPE / "cache/umls_broad_embeddings_sapbert.pkl"),
            # cache-only: a miss returns 0 rather than generating, so no duration LLM call is timed
            BC_CACHE_ONLY="1", WALKER_NO_ONDEMAND="1")

JOBS = {
    "raw_hops":  ([sys.executable, str(PIPE / "build_raw_hops.py")],
              dict(DATASET="_timing_tmp")),   # writes frozen/_timing_tmp/, never a real dataset
    "walker":    ([sys.executable, str(PIPE / "code/eval_kg_walker_full.py")],
                  dict(WALKER_RETRIEVAL_ONLY="1", WALKER_MULTI_AXIS="1", WALKER_NO_MDN="0",
                       WALKER_BC_MODE="overlap", WALKER_OUT_TAG="timing_walker", WALKER_N_WORKERS="3")),
    "tog":       ([sys.executable, str(PIPE / "code/tog_baseline.py")], dict(TOG_OUT_TAG="timing_tog")),
    "hykge":     ([sys.executable, str(PIPE / "code/hykge_baseline.py")], dict(HYKGE_OUT_TAG="timing_hykge")),
    "medrag":    ([sys.executable, str(PIPE / "code/run_medrag_textbook.py")],
                  dict(MEDRAG_PD=str(DD / "durations.json"), MEDRAG_OUT_TAG="timing_medrag")),
}

for name in (sys.argv[1:] or list(JOBS)):
    cmd, extra = JOBS[name]
    env = dict(BASE, **extra)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, cwd=str(PIPE), capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out = {"method": name, "n_questions": N, "seconds_total": round(dt, 2),
           "seconds_per_question": round(dt / N, 3), "returncode": r.returncode,
           "excluded": "per-disease duration LLM generation (cache-only)",
           "tail": (r.stdout or "")[-600:]}
    json.dump(out, open(HERE / f"retrieval_{name}.json", "w"), indent=1)
    print(f"{name:10s} {dt:8.1f}s  = {dt/N:6.2f} s/題   rc={r.returncode}")
