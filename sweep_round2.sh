#!/usr/bin/env bash
# Round 2: N=3 over the walker's own settings. Baselines are NOT re-read — their published N=3
# figures stand.
#
# The comparison that is fully clean is variant-vs-variant: same pool, same reader session, same N.
# Reading a variant against a published baseline additionally spans two sessions, and that gap is
# not small — the pool baseline scored 83.28 on 329 where the published run scored 84.60. Treat
# cross-session gaps under ~1.5pp as unresolved.
#
# Results go to results/param_sweep_n3, kept apart from the N=2 round so the two are never averaged
# together.
#
# The hop-capped variants come from the gold-separation audit rather than from guessing: gold sits
# at hop 0–1 (40/43/20% on 329, 56/36/8% on medbullets) while 74% of distractors are at hop 2,
# where the gold rate is 0.22% against 5.8% at hop 0. raw_1hop is structurally confined to that
# shallow region, so the test is whether the walker's deficit is the hop-2 candidates it spends
# its ten slots on.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=pipeline/logs; mkdir -p "$LOG"
Q=$LOG/_sweep2.txt; : > "$Q"
N=${N_RUNS:-3}
RES=${RESULTS_DIR:-results/param_sweep_n3}
DATASETS=${DATASETS:-"329 medbullets"}

render () {
  local ds="$1"; shift
  if ! env DATASET=$ds METHOD=walker "$@" python3 pipeline/render_from_pool.py > /tmp/_r2.txt 2>&1; then
    echo "  ✗ render failed: $*" | tee -a "$Q"; sed -n '$p' /tmp/_r2.txt; return 1
  fi
  grep -E "變體名稱" /tmp/_r2.txt
}

run () {
  local ds="$1" v="$2"
  [ -f "pipeline/frozen/$ds/$v.json" ] || { echo "  skip $ds/$v (未渲染)" | tee -a "$Q"; return; }
  [ -f "pipeline/$RES/${ds}_${v}_gpt-5.4-mini.json" ] && { echo "  have $ds/$v" | tee -a "$Q"; return; }
  echo "[$(date +%H:%M:%S)] $ds/$v" | tee -a "$Q"
  DATASET=$ds METHOD=$v MODEL=gpt-5.4-mini N_RUNS=$N WORKERS=12 RESULTS_DIR=$RES \
    python3 pipeline/run_reader.py > "$LOG/sweep2_${ds}_${v}.log" 2>&1
  grep -E "^=>" "$LOG/sweep2_${ds}_${v}.log" | tee -a "$Q"
}

for DS in $DATASETS; do
  echo "=== $DS ===" | tee -a "$Q"
  # anchor + round-1 survivors, re-read at N=3 so every number here is on one footing
  render $DS TOP_K=10 LAMBDA=0.3 MU=0.08
  render $DS TOP_K=10 LAMBDA=0   MU=0.08
  render $DS TOP_K=20 LAMBDA=0.3 MU=0.08
  render $DS UTILITY_MODE=criticality W_MAX=0.6 MU=0.08
  render $DS UTILITY_MODE=criticality W_MAX=1.0 MU=0.08
  # new: hop caps, and the two round-1 winners combined
  render $DS MAX_HOP=1 TOP_K=10 LAMBDA=0.3 MU=0.08
  render $DS MAX_HOP=1 TOP_K=20 LAMBDA=0.3 MU=0.08
  render $DS MAX_HOP=1 UTILITY_MODE=criticality W_MAX=0.6 MU=0.08
  render $DS MAX_HOP=0 TOP_K=10 LAMBDA=0.3 MU=0.08
  render $DS TOP_K=20 UTILITY_MODE=criticality W_MAX=0.6 MU=0.08
  # de-confounded: μ scales with (1−w) so w moves only the cos↔bc balance, never the hop penalty
  render $DS UTILITY_MODE=criticality W_MAX=0.6 MU=0.08 HOP_SCALED=1
  render $DS UTILITY_MODE=criticality W_MAX=1.0 MU=0.08 HOP_SCALED=1
  render $DS TOP_K=20 UTILITY_MODE=criticality W_MAX=0.6 MU=0.08 HOP_SCALED=1

  for V in walker__k10_l0.3_m0.08 walker__k10_l0_m0.08 walker__k20_l0.3_m0.08 \
           walker__k10_criticality_w0.6_m0.08 walker__k10_criticality_w1_m0.08 \
           walker__k10_l0.3_m0.08_h1 walker__k20_l0.3_m0.08_h1 \
           walker__k10_criticality_w0.6_m0.08_h1 walker__k10_l0.3_m0.08_h0 \
           walker__k20_criticality_w0.6_m0.08 \
           walker__k10_criticality_w0.6_m0.08_hs walker__k10_criticality_w1_m0.08_hs \
           walker__k20_criticality_w0.6_m0.08_hs; do
    run $DS "$V"
  done
done
echo "[$(date +%H:%M:%S)] === round 2 (N=3) done ===" | tee -a "$Q"
