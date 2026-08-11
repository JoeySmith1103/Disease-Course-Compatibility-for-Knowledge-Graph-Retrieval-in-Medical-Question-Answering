#!/usr/bin/env python3
"""Retrieval step (ORCHESTRATOR ONLY — the Neo4j/graph/score logic is NOT in this file).

This file just points the walker at a dataset and shells out; the actual retrieval chain is:

  build_kg.py  --subprocess-->  pipeline/code/eval_kg_walker_full.py
                                   from kg_walker import walk; walk(...)
                                --> pipeline/code/dkr_policy/kg_walker.py  ==> the real work:
                                      * Neo4j query:  session.run("MATCH (a:Concept {CUI})-[r]-(b:Concept) ...")   (kg_walker:143)
                                      * SCORE UTILITY: score = cos + λ·bc − μ·hop     (kg_walker:5 / walk(); λ=_bc_weight, μ=hop pen, τ=min_score)
                                   from umls_neo4j import get_driver -> bolt://localhost:7687   (umls_neo4j:33)

REQUIRES Neo4j running at bolt://localhost:7687 (creds neo4j/admin; SNOMED-CT `Concept{CUI,name}`).

Usage:  DATASET=329|1273  [WALKER_LLM=gpt-5.4-mini]  python3 pipeline/build_kg.py

To change the SCORE UTILITY: edit the scoring in **pipeline/code/dkr_policy/kg_walker.py** (walk()),
then re-run THIS file (re-retrieves from Neo4j + re-freezes prompts), then pipeline/run_reader.py.
This file only handles dataset wiring + prompt freezing; it does not itself touch Neo4j.
Baselines (medrag/tog/hykge) are separate retrieval scripts, unaffected by a walker score change.
"""
import os, sys, json, subprocess, importlib.util
from pathlib import Path
PIPE = Path(__file__).resolve().parent   # everything resolves inside pipeline/
DS = os.environ.get("DATASET","329"); DD = PIPE/f"datasets/{DS}"
MODEL = os.environ.get("WALKER_LLM","gpt-5.4-mini")
# WALKER_METHOD_NAME lets an ablation (e.g. WALKER_BC_MODE=interval_sample) write to a
# distinct frozen/cache file (walker_interval.json) instead of clobbering the default walker.json.
METHOD_NAME = os.environ.get("WALKER_METHOD_NAME","walker")
TAG = f"pipeline_{DS}_{METHOD_NAME}"

CODE = PIPE/"code"
# NOTE: do NOT put code/dkr_policy on PYTHONPATH — dkr_policy/types.py would shadow stdlib `types`
# at interpreter startup. eval_kg_walker_full.py adds it at RUNTIME (after stdlib is cached) instead.
env = dict(os.environ,
    BENCH_PATH=str(DD/"benchmark.json"), PD_PATH=str(DD/"durations.json"),
    SEEDS_PATH=str(DD/"seeds.json"), SYM_PATH=str(DD/"symptoms.json"),
    QE_PATH=str(DD/"query_entities.json"),   # intents.json no longer used: intent-aware removed
    WALKER_LLM=MODEL, WALKER_MULTI_AXIS="1", WALKER_NO_MDN="0",
    WALKER_N_WORKERS=os.environ.get("WALKER_N_WORKERS","4"), WALKER_OUT_TAG=TAG)
print(f"[build_kg] retrieving walker KG for dataset {DS} (score utility from code/dkr_policy/kg_walker.py) ...")
# cwd=PIPE (not the parent): cwd-relative paths like DKR_MATRIX_PKL="cache/..." must resolve
# inside pipeline/cache, so the folder is self-contained.
subprocess.run([sys.executable, str(CODE/"eval_kg_walker_full.py")], env=env, check=True, cwd=str(PIPE))

# freeze prompts from the fresh output
# eval_kg_walker_full writes its checkpoint to pipeline/cache (its own CACHE=ROOT_pipeline/cache),
# so the freeze must read from there — NOT new_duration_spectrum/cache.
out_file = PIPE/f"cache/{TAG}__{MODEL.replace('/','_')}.json"
spec = importlib.util.spec_from_file_location("P", PIPE/"prompts.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
def days_to_phrase(d):
    if isinstance(d,(list,tuple)):
        ph=[days_to_phrase(x) for x in d]; seen=[]
        for p in ph:
            if p not in seen: seen.append(p)
        return " / ".join(seen) if len(seen)>1 else seen[0]
    if d<1: return f"{int(d*24)} hours"
    if d<14: return f"{int(d)} days"
    if d<60: return f"{int(d/7)} weeks"
    if d<365: return f"{int(d/30)} months"
    return f"{int(d/365)} years"
def ob(o): return "\n".join(f"  {k}. {v}" for k,v in sorted(o.items()))
B={x["uid"]:x for x in json.load(open(DD/"benchmark.json"))}
recs=json.load(open(out_file)); recs=recs.get("results",recs)
items=[]
for r in recs:
    it=B[r["uid"]]
    # walker_kg_nodur = no usable duration, retrieved anyway with bc off (cos − μ·hop)
    if r.get("route") in ("walker_kg","walker_kg_nodur") and r.get("kg_block"):
        pd_days=r.get("patient_days")
        dur_str=days_to_phrase(pd_days) if pd_days else P.NO_DURATION_STR
        pr=P.WALKER.format(question=it["question"],options_block=ob(it["options"]),
                           patient_dur_str=dur_str,kg_block=r["kg_block"])
    else:
        pr=P.NO_KG.format(question=it["question"],options_block=ob(it["options"]))
    items.append({"uid":r["uid"],"gold":r["gold"],"route":r.get("route"),"kg_block":r.get("kg_block",""),"prompt":pr})
(PIPE/f"frozen/{DS}").mkdir(parents=True, exist_ok=True)
json.dump({"method":METHOD_NAME,"dataset":DS,"n":len(items),"items":items},
          open(PIPE/f"frozen/{DS}/{METHOD_NAME}.json","w"), indent=1)
print(f"[build_kg] froze {len(items)} prompts -> pipeline/frozen/{DS}/{METHOD_NAME}.json")
print(f"[build_kg] now run:  DATASET={DS} METHOD={METHOD_NAME} N_RUNS=5 python3 pipeline/run_reader.py")
