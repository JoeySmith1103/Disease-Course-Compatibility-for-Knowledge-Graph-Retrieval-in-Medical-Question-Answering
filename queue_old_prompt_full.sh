#!/usr/bin/env bash
# Full old-prompt A/B at N=3, queued behind everything else.
#
# peek_old_prompt.sh answers "is it worth measuring"; this answers "how big is it". Same frozen
# evidence, original wording, three runs, every dataset that has KG methods frozen.
#
# medbullets is skipped if the peek already wrote N=3 results — re-reading the same frozen prompts
# costs API credit and yields the same number.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=pipeline/logs; mkdir -p "$LOG"
Q=$LOG/_oldprompt_full.txt; : > "$Q"

# anchored — an unanchored pattern also matches waiting shells that merely mention the reader in
# their own script text, which deadlocks every waiter against every other waiter
wait_for_slot () {
  while pgrep -f "^python3 pipeline/run_reader\.py" > /dev/null \
     || pgrep -f "peek_old_prompt\.sh|queue_readers\.sh" > /dev/null; do sleep 60; done
}

echo "[$(date +%H:%M:%S)] waiting for the reader queue and the peek to clear…" | tee -a "$Q"
wait_for_slot
echo "[$(date +%H:%M:%S)] slot free" | tee -a "$Q"

# 329 and 1273 are deliberately out of scope here — they are settled results, and re-reading them
# under a second prompt would produce a competing number for a dataset whose figures are already
# fixed. The prompt question is being answered on the NEW datasets only.
for DS in ${DATASETS:-medbullets mmlu}; do
  for M in walker walker_interval raw_1hop raw_2hop tog hykge; do
    f="pipeline/frozen/$DS/$M.json"
    [ -f "$f" ] || continue
    out="pipeline/results/old_prompt/${DS}_${M}_gpt-5.4-mini.json"
    if [ -f "$out" ] && [ "$(python3 -c "import json,sys;print(len(json.load(open('$out'))['runs_correct']))" 2>/dev/null)" = "3" ]; then
      echo "[$(date +%H:%M:%S)] have N=3 already: $DS/$M" | tee -a "$Q"; continue
    fi
    echo "[$(date +%H:%M:%S)] START $DS/$M (old prompt, N=3)" | tee -a "$Q"
    DATASET=$DS METHOD=$M MODEL=gpt-5.4-mini N_RUNS=3 WORKERS=12 \
      RESULTS_DIR=results/old_prompt \
      python3 pipeline/run_reader.py > "$LOG/old_${DS}_${M}.log" 2>&1
    tail -1 "$LOG/old_${DS}_${M}.log" | tee -a "$Q"
  done
done
echo "[$(date +%H:%M:%S)] === old-prompt full A/B done ===" | tee -a "$Q"
