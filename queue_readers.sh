#!/usr/bin/env bash
# Reader queue: everything that still needs a reader pass, in the order it can legally run.
#
# Serialised on purpose — the reader is API-bound and two passes at once just split the same rate
# limit, which shows up as parse failures rather than as speed. Retrieval (stage A) runs in a
# different process and is allowed to overlap, since it is Neo4j/embedding-bound.
#
# vanilla/cot are read from the ORIGINAL frozen files, not from a __revised re-render: the revised
# template announces knowledge-graph evidence, so applying it to a no-KG control would compare a
# method against a prompt that promises evidence and shows none.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=logs; mkdir -p "$LOG"
N=${N_RUNS:-3}
Q=$LOG/_reader_queue.txt; : > "$Q"

read_one () {                 # read_one <dataset> <method>
  local ds="$1" m="$2" tag="${1}_${2}"
  [ -f "frozen/$ds/$m.json" ] || { echo "[$(date +%H:%M:%S)] SKIP $tag (not frozen)" | tee -a "$Q"; return; }
  echo "[$(date +%H:%M:%S)] START $tag" | tee -a "$Q"
  DATASET=$ds METHOD=$m MODEL=gpt-5.4-mini N_RUNS=$N WORKERS=12 \
    RESULTS_DIR=results/revised_prompt \
    python3 run_reader.py > "$LOG/read_$tag.log" 2>&1
  tail -1 "$LOG/read_$tag.log" | tee -a "$Q"
}

# ── 1. MedBullets no-KG controls (frozen already, can run right now) ─────────
read_one medbullets vanilla
read_one medbullets cot

# ── 2. MMLU — wait for stage A to finish writing every frozen file ───────────
echo "[$(date +%H:%M:%S)] waiting for stage A (mmlu) to finish…" | tee -a "$Q"
while pgrep -f "run_stage_a.sh mmlu" > /dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] stage A (mmlu) done" | tee -a "$Q"

bash rerender_all_kg.sh mmlu >> "$LOG/mmlu_rerender.log" 2>&1
echo "[$(date +%H:%M:%S)] rerendered mmlu → __revised" | tee -a "$Q"

read_one mmlu vanilla
read_one mmlu cot
for m in walker walker_interval raw_1hop raw_2hop tog hykge; do read_one mmlu "${m}__revised"; done

echo "[$(date +%H:%M:%S)] === reader queue done ===" | tee -a "$Q"
