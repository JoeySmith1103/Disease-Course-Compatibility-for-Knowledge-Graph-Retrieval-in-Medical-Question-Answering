#!/usr/bin/env python3
"""Add one reader pass to an existing result file instead of re-running the whole thing.

Two reasons, and the second is the important one.

BUDGET. Promoting a screened variant from N=1 to N=3 used to mean three fresh passes, throwing
away the pass already paid for. This appends, so N=1 → N=3 costs two passes.

VARIANCE. run_reader.py's N passes run back-to-back inside one invocation, and they come out far
too similar to each other. Measured on medbullets/walker__k10_jmagnitude_n0.15_os with a
byte-identical prompt (verified 308/308):

    19:05, one pass      255 correct
    19:12, three passes  244 / 243 / 243     ← spread of 1 within the call

Three consecutive passes disagreed by one question; two calls seven minutes apart disagreed by
twelve. The same pattern shows on raw_1hop/medbullets across days: 81.06 / 78.25 / 79.55. So the
std printed from three consecutive passes is not the uncertainty of the measurement — it is the
uncertainty of one short time window, and it is roughly an order of magnitude too small.

Passes accumulated through this script are spaced by whenever it is invoked, so their spread
reflects the between-call variance that actually limits what can be concluded. `spread_note` in
the output records which passes were consecutive and which were separate calls.

Usage:
  DATASET=medbullets METHOD=walker__k10_jmagnitude_n0.15_os RESULTS_DIR=results/p2 \
    python3 pipeline/append_run.py
"""
import json, os, statistics, subprocess, sys, time
from pathlib import Path

PIPE = Path(__file__).resolve().parent
DS    = os.environ.get("DATASET", "medbullets")
METH  = os.environ.get("METHOD")
MODEL = os.environ.get("MODEL", "gpt-5.4-mini")
RES   = PIPE / os.environ.get("RESULTS_DIR", "results")
if not METH:
    sys.exit("need METHOD=<frozen variant name>")

target = RES / f"{DS}_{METH}_{MODEL.replace('/','_')}.json"
tmp = RES / f".append_{DS}_{METH}.json"

env = dict(os.environ, DATASET=DS, METHOD=METH, MODEL=MODEL, N_RUNS="1",
           RESULTS_DIR=str(tmp.parent.relative_to(PIPE)))
# run_reader writes to RESULTS_DIR/<ds>_<method>_<model>.json; stage it aside so an existing file
# is never clobbered before the merge succeeds
staging = RES / "_staging"
staging.mkdir(parents=True, exist_ok=True)
env["RESULTS_DIR"] = str(staging.relative_to(PIPE))
r = subprocess.run([sys.executable, str(PIPE / "run_reader.py")], env=env, cwd=str(PIPE),
                   capture_output=True, text=True)
if r.returncode != 0:
    sys.exit(f"reader failed:\n{r.stdout[-1500:]}\n{r.stderr[-800:]}")
new_path = staging / f"{DS}_{METH}_{MODEL.replace('/','_')}.json"
new = json.load(open(new_path))

if target.exists():
    old = json.load(open(target))
    runs = old["runs"] + [{"run": len(old["runs"]) + 1, "results": new["runs"][0]["results"]}]
    correct = old["runs_correct"] + new["runs_correct"]
    unpar = old["runs_unparseable"] + new["runs_unparseable"]
    note = old.get("spread_note", []) + [f"call {len(old.get('spread_note', [])) + 1}: 1 pass"]
else:
    runs, correct, unpar = new["runs"], new["runs_correct"], new["runs_unparseable"]
    note = ["call 1: 1 pass"]

n = new["n"]
out = {"dataset": DS, "method": METH, "model": MODEL, "n": n,
       "runs_correct": correct, "mean_correct": statistics.fmean(correct),
       "std_correct": statistics.pstdev(correct) if len(correct) > 1 else 0.0,
       "mean_acc": 100 * statistics.fmean(correct) / n,
       "std_acc": 100 * (statistics.pstdev(correct) if len(correct) > 1 else 0.0) / n,
       "runs_unparseable": unpar, "mean_unparseable": statistics.fmean(unpar),
       # each pass came from its own invocation, so this spread is the between-call variance —
       # the one that actually bounds what can be claimed
       "spread_note": note, "passes_are_separate_calls": True,
       "runs": runs}
json.dump(out, open(target, "w"), indent=1)
new_path.unlink(missing_ok=True)
print(f"{DS}/{METH}: {len(correct)} 次獨立呼叫  {correct}  "
      f"= {out['mean_acc']:.2f} ± {out['std_acc']:.2f}%")
