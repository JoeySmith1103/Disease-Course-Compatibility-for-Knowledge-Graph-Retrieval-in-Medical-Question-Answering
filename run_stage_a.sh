#!/usr/bin/env bash
# Stage A for a new dataset: retrieve with every method and freeze one file per method.
#
# Runs the retrieval steps SEQUENTIALLY on purpose. Each walker worker holds ~4 GB of SapBERT
# matrix, and two retrieval jobs at once has previously exhausted a 38 GB machine and forced a
# reboot. Baselines are also serialised so a Neo4j timeout in one cannot be blamed on contention
# from another.
#
# Every method writes its own frozen/<ds>/<method>.json, and inside each of those the kg_block is
# stored separately from the prompt — that is what makes later prompt-tuning free (rerender_prompts.py
# rebuilds wording from the same evidence without touching Neo4j).
#
# A failing baseline does not abort the run: the step is logged as FAILED and the script moves on,
# so one broken method cannot cost you the whole retrieval pass.
#
# Usage:  bash pipeline/run_stage_a.sh medbullets
#         bash pipeline/run_stage_a.sh mmlu
# Logs:   pipeline/logs/<ds>_<step>.log       Progress: pipeline/logs/<ds>_STATUS.txt

set -u
DS="${1:?usage: run_stage_a.sh <dataset>}"
MODEL="${WALKER_LLM:-gpt-5.4-mini}"
NW="${WALKER_N_WORKERS:-3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="pipeline/logs"; mkdir -p "$LOG"
DD="pipeline/datasets/$DS"
STATUS="$LOG/${DS}_STATUS.txt"; : > "$STATUS"

step () {                     # step <name> <command...>
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$STATUS"
  if "$@" > "$LOG/${DS}_${name}.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK    $name" | tee -a "$STATUS"
  else
    echo "[$(date +%H:%M:%S)] FAILED $name  (see $LOG/${DS}_${name}.log)" | tee -a "$STATUS"
  fi
}

# ── preflight: the walker cannot run without these, and failing here is cheaper than failing
#    40 minutes into a retrieval pass
for f in benchmark durations seeds symptoms query_entities; do
  [ -s "$DD/$f.json" ] || { echo "MISSING $DD/$f.json — run prepare_dataset_inputs.py first"; exit 1; }
done
python3 - <<'PY' || exit 1
import sys; sys.path.insert(0,'pipeline/code'); sys.path.insert(0,'pipeline/code/dkr_policy')
from umls_neo4j import get_driver
n = get_driver().session().run('MATCH (c:Concept) RETURN count(c) AS n').single()['n']
print(f'Neo4j OK: {n} concepts')
assert n > 400000, 'concept count too low — wrong DB or a half-open tunnel'
PY

echo "=== stage A for $DS (model=$MODEL, workers=$NW) ===" | tee -a "$STATUS"

# ── no Neo4j ────────────────────────────────────────────────────────────────
DATASET=$DS step prompt_only python3 pipeline/build_prompt_only.py

# ── walker + its ablation (Neo4j) ───────────────────────────────────────────
DATASET=$DS WALKER_BC_MODE=overlap WALKER_METHOD_NAME=walker \
  WALKER_RETRIEVAL_ONLY=1 WALKER_N_WORKERS=$NW WALKER_LLM=$MODEL \
  step walker python3 pipeline/build_kg.py

DATASET=$DS WALKER_BC_MODE=interval_sample WALKER_METHOD_NAME=walker_interval \
  WALKER_RETRIEVAL_ONLY=1 WALKER_N_WORKERS=$NW WALKER_LLM=$MODEL \
  step walker_interval python3 pipeline/build_kg.py

# ── raw dumps (Neo4j) ───────────────────────────────────────────────────────
DATASET=$DS step raw_hops python3 pipeline/build_raw_hops.py

# ── KG-RAG baselines (Neo4j for ToG/HyKGE, not for MedRAG) ──────────────────
BENCH_PATH=$DD/benchmark.json SEEDS_PATH=$DD/seeds.json SYM_PATH=$DD/symptoms.json \
  TOG_LLM=$MODEL KG_OUT_TAG=$DS TOG_WORKERS=4 \
  step tog_retrieve python3 pipeline/code/tog_baseline.py
DATASET=$DS METHOD=tog SRC=cache/tog_baseline_${DS}__${MODEL}.json \
  step tog_freeze python3 pipeline/freeze_baseline.py

BENCH_PATH=$DD/benchmark.json SEEDS_PATH=$DD/seeds.json SYM_PATH=$DD/symptoms.json \
  HYKGE_LLM=$MODEL KG_OUT_TAG=$DS HYKGE_WORKERS=2 \
  step hykge_retrieve python3 pipeline/code/hykge_baseline.py
DATASET=$DS METHOD=hykge SRC=cache/hykge_baseline_${DS}__${MODEL}.json \
  step hykge_freeze python3 pipeline/freeze_baseline.py

# MedRAG is ON HOLD. Three things have to be settled first, and none of them is about runtime:
#   1. corpus provenance — the index is MedQA's own 18 textbooks (cache/medqa_textbook_chunks.pkl).
#      Reusing it for MedBullets/MMLU is defensible (all USMLE-style, and the MedRAG paper reuses
#      one corpus across benchmarks) but it is not a corpus built for these datasets, so the choice
#      has to be stated rather than inherited silently.
#   2. run_medrag_textbook.py:17/82 still import call_llm from `spectrum_textbook` via
#      pipeline/scripts — that path no longer exists (consolidated into code/llm_client.py).
#   3. MEDRAG_PD defaults to a bench340 duration file that is not in cache.
# Re-enable by restoring the two lines below once those are resolved.
#   BENCH_PATH=$DD/benchmark.json MEDRAG_OUT_TAG=$DS \
#     step medrag_retrieve python3 pipeline/code/run_medrag_textbook.py
#   DATASET=$DS METHOD=medrag SRC=cache/${DS}_medrag_textbook_k32__${MODEL}.json \
#     step medrag_freeze python3 pipeline/freeze_baseline.py

# orphaned ProcessPool workers keep burning LLM credit; WALKER_OUT_TAG is an env var so pgrep
# on the tag misses them — kill by script name instead
pkill -9 -f eval_kg_walker_full.py 2>/dev/null

echo "=== done: $DS ===" | tee -a "$STATUS"
python3 - "$DS" <<'PY' | tee -a "$STATUS"
import json, sys, glob, os
ds = sys.argv[1]
print(f"\nfrozen/{ds}/ :")
for f in sorted(glob.glob(f"pipeline/frozen/{ds}/*.json")):
    # the folder can also hold hand-placed files (temporal_critical.json, archives); a frozen
    # method is identified by carrying an "items" list, not by living in this directory
    try: d = json.load(open(f))
    except Exception: continue
    if not isinstance(d, dict) or "items" not in d: continue
    items = d["items"]
    kg = sum(1 for i in items if (i.get("kg_block") or "").strip())
    print(f"  {os.path.basename(f):24s} n={len(items):4d}  有 kg_block {kg:4d}")
PY
