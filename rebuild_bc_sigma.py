#!/usr/bin/env python3
"""Recompute every candidate's bc at a given patient-duration sigma, from the LLM duration cache.

sigma_p is the width of the log-normal placed on the patient's elapsed time: the patient is
"t days in" is read as N(log t, sigma_p). It was a default argument in bc_for_cui -- 0.30, meaning
roughly +-35% on the stated duration -- and like the hop penalty it has no derivation behind it.

It cannot be swept by re-ranking: bc is computed during the walk and stored, so a different sigma
needs the numbers rebuilt. That is cheap offline (the cache holds the per-disease parameters) but
it does change the pool, so each sigma gets its own file rather than overwriting.

The rebuild recomputes bc for EVERY disease candidate from the LLM cache, including the ~18% whose
stored value came from the walk's MDN fallback. Those move to the LLM path here, so the sigma=0.30
rebuild is the reference point for the sweep rather than the shipped pool itself.

Usage:  DATASET=medbullets SIGMA_P=0.6 python3 pipeline/rebuild_bc_sigma.py
"""
import json, os, sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "code")); sys.path.insert(0, str(PIPE / "code" / "dkr_policy"))
os.environ.setdefault("DKR_MATRIX_PKL", str(PIPE / "cache/umls_broad_embeddings_sapbert.pkl"))
os.environ.setdefault("WALKER_BC_MODE", "overlap")
from bc_llm_direct import bc_for_cui

DS = os.environ.get("DATASET", "medbullets")
SIG = float(os.environ.get("SIGMA_P", "0.30"))
SRC = os.environ.get("SRC_POOL", "union_qall")
tag = f"{SRC}_s{SIG:g}"
pd1 = lambda v: (float(v[0]) if isinstance(v, (list, tuple)) and v
                 else (float(v) if isinstance(v, (int, float)) else None))

out = PIPE / f"pool/{DS}/{tag}.jsonl"
hit = miss = changed = 0
with open(out, "w") as fh:
    for line in open(PIPE / f"pool/{DS}/{SRC}.jsonl"):
        r = json.loads(line)
        t = pd1(r.get("patient_days"))
        for c in r["candidates"]:
            if c.get("role") != "disease" or not t:
                c["bc"] = 0.0; continue
            v = bc_for_cui(c["cui"], t, sigma_p=SIG)
            if v > 0: hit += 1
            else: miss += 1
            if abs(v - c.get("bc", 0.0)) > 1e-6: changed += 1
            c["bc"] = round(v, 6)
        for c in r["candidates"]:
            c["score"] = c.get("cos", 0.0) + 0.3 * c["bc"] - 0.08 * c.get("hop", 0)
        r["candidates"].sort(key=lambda c: -c["score"])
        for i, c in enumerate(r["candidates"], 1): c["rank"] = i
        r["params"] = {**(r.get("params") or {}), "sigma_p": SIG, "bc_source": "llm_cache"}
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{DS} sigma_p={SIG} -> pool/{DS}/{tag}.jsonl   cache 命中 {hit}，未命中 {miss}"
      f"（{100*hit/max(hit+miss,1):.0f}%），bc 改變 {changed}")
