#!/usr/bin/env bash
# Utility-parameter sweep over the stored pools.
#
# The baseline is rendered FROM THE POOL too, not taken from the published frozen numbers. The pool
# reproduces frozen exactly on 329 and mmlu but differs on 3 medbullets questions (a duration-cache
# entry appeared after that run), so comparing a pool variant against a published figure would
# charge 1% of drift to the variant. Same substrate on both sides or the comparison is not clean.
#
# N=2 per the current convention. Two runs cannot resolve a 1-2pp difference — treat anything
# inside ±1.5pp as unresolved and re-run the survivors at higher N.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=pipeline/logs; mkdir -p "$LOG"
Q=$LOG/_sweep.txt; : > "$Q"
N=${N_RUNS:-2}
DATASETS=${DATASETS:-"329 medbullets"}

render () {   # render <ds> <env assignments...>
  # `env` is required: bash only treats VAR=val as an assignment when it is a LITERAL word at the
  # start of the command. Coming out of "$@" it is just a word, so bash tried to execute `TOP_K=10`
  # as the command name and every render silently failed.
  local ds="$1"; shift
  if ! env DATASET=$ds METHOD=walker "$@" python3 pipeline/render_from_pool.py > /tmp/_r.txt 2>&1; then
    echo "  ✗ render failed: $*"; sed -n '$p' /tmp/_r.txt; return 1
  fi
  grep -E "變體名稱|斷點" /tmp/_r.txt
}

run () {      # run <ds> <variant>
  local ds="$1" v="$2"
  [ -f "pipeline/frozen/$ds/$v.json" ] || { echo "  skip $ds/$v (未渲染)" | tee -a "$Q"; return; }
  echo "[$(date +%H:%M:%S)] $ds/$v" | tee -a "$Q"
  DATASET=$ds METHOD=$v MODEL=gpt-5.4-mini N_RUNS=$N WORKERS=12 \
    RESULTS_DIR=results/param_sweep python3 pipeline/run_reader.py \
    > "$LOG/sweep_${ds}_${v}.log" 2>&1
  grep -E "^=>" "$LOG/sweep_${ds}_${v}.log" | tee -a "$Q"
}

for DS in $DATASETS; do
  echo "=== render $DS ===" | tee -a "$Q"
  render $DS TOP_K=10 LAMBDA=0.3 MU=0.08                       # pool baseline
  render $DS TOP_K=10 LAMBDA=0.3 MU=0                          # hop ablation: penalty off
  render $DS TOP_K=10 LAMBDA=0.3 MU=0.16                       # hop ablation: doubled
  render $DS TOP_K=10 LAMBDA=0   MU=0.08                       # bc ablation
  render $DS TOP_K=20 LAMBDA=0.3 MU=0.08                       # more evidence
  render $DS UTILITY_MODE=criticality W_MAX=0.6 MU=0.08        # variant A
  render $DS UTILITY_MODE=criticality W_MAX=1.0 MU=0.08        # variant A, uncapped
  render $DS UTILITY_MODE=adaptive    W_MAX=0.4 MU=0.08        # variant B
  render $DS UTILITY_MODE=adaptive    W_MAX=0.6 MU=0.08        # variant B, stronger

  for V in walker__k10_l0.3_m0.08 walker__k10_l0.3_m0 walker__k10_l0.3_m0.16 \
           walker__k10_l0_m0.08 walker__k20_l0.3_m0.08 \
           walker__k10_criticality_w0.6_m0.08 walker__k10_criticality_w1_m0.08 \
           walker__k10_adaptive_w0.4_m0.08 walker__k10_adaptive_w0.6_m0.08; do
    run $DS "$V"
  done
done
echo "[$(date +%H:%M:%S)] === sweep done ===" | tee -a "$Q"
