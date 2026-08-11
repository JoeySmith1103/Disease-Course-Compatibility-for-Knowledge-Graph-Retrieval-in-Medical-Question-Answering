#!/usr/bin/env python3
"""Answer step: run the reader LLM on FROZEN per-question prompts (frozen/<dataset>/<method>.json),
N times, and report mean ± std correct-count. No retrieval here — retrieval is frozen upstream.

Usage:
  DATASET=329|1273  METHOD=vanilla|cot|medrag|tog|hykge|walker  MODEL=gpt-5.4-mini  N_RUNS=5 \
  python3 pipeline/run_reader.py
Output: pipeline/results/<dataset>_<method>_<model>.json  (per-run counts + mean/std)
"""
import json, re, os, sys, statistics, importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE/"code"))
sys.path.insert(0, str(PIPE/"code"/"dkr_policy"))
from llm_client import call_llm

_spec = importlib.util.spec_from_file_location("P", PIPE/"prompts.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)
sys.path.insert(0, str(PIPE))
from metrics import compute_metrics, aggregate, format_report, _labels_from_options

DATASET = os.environ.get("DATASET","329")
METHOD  = os.environ.get("METHOD","walker")
MODEL   = os.environ.get("MODEL","gpt-5.4-mini")
N_RUNS  = int(os.environ.get("N_RUNS","1"))
WORKERS = int(os.environ.get("WORKERS","12"))

from answer_extract import extract_letter

items = json.load(open(PIPE/f"frozen/{DATASET}/{METHOD}.json"))["items"]
# Fixed label set from the dataset's own options. Deriving it from what a run happened to predict
# would give each method a different denominator, and macro numbers exist to be compared across
# methods.
BENCH  = json.load(open(PIPE/f"datasets/{DATASET}/benchmark.json"))
OPTS   = {b["uid"]: b.get("options", {}) for b in BENCH}
LABELS = _labels_from_options(BENCH)
runs=[]; per_run=[]
def _evidence_lines(t):
    # content of each evidence line, stripped of list markers ("  1. " / "- "), for the canary.
    return {re.sub(r"^\s*(?:\d+\.|[-*])\s*", "", ln).strip()
            for ln in (t or "").splitlines() if ln.strip()}

def one(it):
    kg = it.get("kg_block") or ""
    # prompt surgery is centralised in prompts.py; big models keep <a></a>, lose the path legend
    prompt = P.to_big_model_prompt(it["prompt"])
    # Canary: evidence must actually reach the prompt. Compare by line CONTENT, not verbatim
    # substring — some baselines (HyKGE) number kg_block ("  1. ") but bullet it in the prompt
    # ("- "). A real drop still trips this; pure formatting does not.
    assert (not kg) or _evidence_lines(kg) <= _evidence_lines(prompt), \
        f"BUG: kg_block evidence missing from prompt for {it['uid']}"
    raw = call_llm(prompt, model=MODEL)
    # options let the extractor reject a letter the question never offered, and accept an exact
    # option string as a last resort
    pred = extract_letter(raw, OPTS.get(it["uid"]))
    # ALWAYS persist the exact prompt sent and the raw output, not just the verdict.
    return {"uid":it["uid"],"gold":it["gold"],"predicted":pred,"is_correct":pred==it["gold"],
            "route":it.get("route"),"kg_block":kg,"prompt":prompt,"raw_response":raw}
unpar=[]                      # per-run count of answers extract_letter() could not parse
for run in range(N_RUNS):
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        recs=list(ex.map(one, items))
    c=sum(1 for r in recs if r["is_correct"]); runs.append(c); per_run.append(recs)
    none=sum(1 for r in recs if r["predicted"] is None); unpar.append(none)
    print(f"  run {run+1}/{N_RUNS}: {c}/{len(items)} = {100*c/len(items):.2f}%  unparseable={none}", flush=True)
mean=statistics.fmean(runs); sd=statistics.pstdev(runs) if len(runs)>1 else 0.0
# Unparseable answers are scored WRONG, so a high count silently deflates accuracy — surface it
# in the summary line, not only per-run (a throttled run can look like a worse model).
print(f"=> {METHOD}/{DATASET}/{MODEL}: {mean:.1f} ± {sd:.1f} correct / {len(items)} "
      f"= {100*mean/len(items):.2f} ± {100*sd/len(items):.2f}%  "
      f"| unparseable {unpar} (mean {statistics.fmean(unpar):.1f} = {100*statistics.fmean(unpar)/len(items):.2f}%)")

# precision/recall over ANSWER LETTERS — a position-bias measure, not diagnostic quality, and not
# the retrieval recall the method is argued on. See metrics.py.
run_metrics = [compute_metrics(recs, labels=LABELS) for recs in per_run]
METRICS = aggregate(run_metrics)
print(format_report(METRICS))
RES = PIPE/os.environ.get("RESULTS_DIR", "results")   # keep new rounds out of the old results
RES.mkdir(parents=True, exist_ok=True)
json.dump({"dataset":DATASET,"method":METHOD,"model":MODEL,"n":len(items),
           "runs_correct":runs,"mean_correct":mean,"std_correct":sd,
           "mean_acc":100*mean/len(items),"std_acc":100*sd/len(items),
           # parse failures are counted as wrong -> persist them so a number can be audited
           # from the JSON alone, without digging through console logs or per-question records
           "runs_unparseable":unpar,"mean_unparseable":statistics.fmean(unpar),
           # letter-level precision/recall, aggregated over runs, plus the per-run detail so a
           # per-class breakdown can be recovered without re-scoring
           "metrics":METRICS,"metrics_per_run":run_metrics,
           # full per-run trace: prompt + raw output for every question, every run
           "runs":[{"run":i+1,"results":recs} for i,recs in enumerate(per_run)]},
          open(RES/f"{DATASET}_{METHOD}_{MODEL.replace('/','_')}.json","w"), indent=1)
