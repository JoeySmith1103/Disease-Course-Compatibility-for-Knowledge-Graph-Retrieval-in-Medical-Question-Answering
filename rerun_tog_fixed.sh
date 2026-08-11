#!/usr/bin/env bash
# Re-read ToG after the evidence fix.
#
# tog_baseline.py wrote a scored ENTITY ranking into kg_block while building prompt_full from the
# CHAINS. Replaying the frozen prompt was unaffected, but every __revised variant is rebuilt FROM
# kg_block — so "tog__revised" was a bare entity list wearing ToG's name, and its numbers do not
# describe Think-on-Graph. The freeze now picks the field the prompt actually contains, so these
# runs replace the invalid ones.
#
# No retrieval is redone: chains_full was already persisted in cache, so the fix cost nothing but
# a re-freeze.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=logs; mkdir -p "$LOG"
Q=$LOG/_tog_refix.txt; : > "$Q"

while pgrep -f "^python3 run_reader\.py" > /dev/null; do sleep 20; done

run () {                      # run <ds> <method> <n> <results_dir>
  echo "[$(date +%H:%M:%S)] START $1/$2 N=$3 -> $4" | tee -a "$Q"
  DATASET=$1 METHOD=$2 MODEL=gpt-5.4-mini N_RUNS=$3 WORKERS=12 RESULTS_DIR=$4 \
    python3 run_reader.py > "$LOG/tog_refix_${1}_${2}.log" 2>&1
  tail -1 "$LOG/tog_refix_${1}_${2}.log" | tee -a "$Q"
}

run medbullets tog          1 results/old_prompt        # the cell the canary blocked in the peek
run 329        tog__revised 3 results/revised_prompt    # replaces the entity-list run
run medbullets tog__revised 3 results/revised_prompt    # replaces the entity-list run
echo "[$(date +%H:%M:%S)] === tog refix done ===" | tee -a "$Q"
