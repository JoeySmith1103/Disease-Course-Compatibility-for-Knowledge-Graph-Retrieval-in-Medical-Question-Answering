#!/usr/bin/env bash
# Quick look: same frozen evidence, ORIGINAL prompt wording, one run each.
#
# The A/B this answers: on 329 every KG method collapsed into a 79.6–82.0% band under the revised
# template. Either the methods really are indistinguishable, or the template flattens them. Since
# kg_block is frozen separately from prompt, <method>.json and <method>__revised.json carry
# BYTE-IDENTICAL evidence and differ only in wording — so a gap between the two columns is a prompt
# effect and nothing else.
#
# N_RUNS=1 by design: this is a peek, not the measurement. Run-to-run spread on 308 questions is
# ~1pp, so only a difference well clear of that means anything here.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS="${1:-medbullets}"
N="${N_RUNS:-1}"
LOG=pipeline/logs; mkdir -p "$LOG"
Q=$LOG/_oldprompt_${DS}.txt; : > "$Q"

# Do not overlap with the main reader queue — the reader is API-bound and two passes at once buy
# nothing but throttling.
#
# The regex is ANCHORED: an unanchored `pgrep -f run_reader.py` also matches any waiting shell
# whose own script text mentions run_reader.py, so a queue that is merely *waiting* to call the
# reader reads as a reader already running, and every waiter deadlocks on every other waiter. The
# real process is exactly `python3 pipeline/run_reader.py` (env assignments are not in argv).
while pgrep -f "^python3 pipeline/run_reader\.py" > /dev/null; do sleep 20; done

for m in walker walker_interval raw_1hop raw_2hop tog hykge; do
  [ -f "pipeline/frozen/$DS/$m.json" ] || { echo "SKIP $m" | tee -a "$Q"; continue; }
  echo "[$(date +%H:%M:%S)] START $DS/$m (old prompt)" | tee -a "$Q"
  DATASET=$DS METHOD=$m MODEL=gpt-5.4-mini N_RUNS=$N WORKERS=12 \
    RESULTS_DIR=results/old_prompt \
    python3 pipeline/run_reader.py > "$LOG/old_${DS}_${m}.log" 2>&1
  tail -1 "$LOG/old_${DS}_${m}.log" | tee -a "$Q"
done
echo "[$(date +%H:%M:%S)] === old-prompt peek done: $DS ===" | tee -a "$Q"
