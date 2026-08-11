#!/usr/bin/env python3
"""Answer step for small/medium open models using the block-structured prompt.

Reads frozen/<dataset>/<method>.json and does no retrieval. The evidence and method stay frozen;
only the final answer-format instruction is replaced by prompts.to_small_model_prompt().

Usage:
  DATASET=329 MODEL=together:Qwen/Qwen3.5-9B RUN=1 WORKERS=6 \
    METHODS=vanilla,cot,raw_1hop,raw_2hop,medrag,tog,hykge,walker,walker_interval \
    python3 run_reader_block.py
"""
import json, os, re, sys, importlib.util, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(PIPE / "code"))
sys.path.insert(0, str(PIPE / "code" / "dkr_policy"))
from llm_client import call_llm
from answer_extract import extract_letter
from metrics import compute_metrics, _labels_from_options

spec = importlib.util.spec_from_file_location("P", PIPE / "prompts.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

DATASET = os.environ.get("DATASET", "1273")
MODEL   = os.environ.get("MODEL", "together:Qwen/Qwen3.5-9B")
RUN     = os.environ.get("RUN", "1")
WORKERS = int(os.environ.get("WORKERS", "6"))
METHODS = [m for m in os.environ.get("METHODS", "vanilla,cot,raw_1hop,raw_2hop,medrag,tog,hykge,walker,walker_interval").split(",") if m]
SUMMARY_KG_METHODS = set(x for x in os.environ.get("SUMMARY_KG_METHODS", "tog,hykge").split(",") if x)

MODEL_TAG = MODEL.split("/")[-1].replace(":", "_")
RES = PIPE / os.environ.get("RESULTS_DIR", "results")
RES.mkdir(parents=True, exist_ok=True)

BENCH = json.load(open(PIPE / f"datasets/{DATASET}/benchmark.json"))
OPTS = {b["uid"]: b.get("options", {}) for b in BENCH}
LABELS = _labels_from_options(BENCH)


def _evidence_lines(t):
    return {re.sub(r"^\s*(?:\d+\.|[-*])\s*", "", ln).strip()
            for ln in (t or "").splitlines() if ln.strip()}


def _base_method(method):
    return method.split("__", 1)[0]


for method in METHODS:
    fp = PIPE / f"frozen/{DATASET}/{method}.json"
    if not fp.exists():
        print(f"[{method}] no frozen file, skip", flush=True)
        continue
    items = json.load(open(fp))["items"]
    out_path = RES / f"{DATASET}_{method}_{MODEL_TAG}_block_run{RUN}.json"

    done = {}
    if out_path.exists():
        try:
            done = {r["uid"]: r for r in json.load(open(out_path)).get("results", [])}
        except Exception:
            done = {}
    todo = [it for it in items if it["uid"] not in done]
    results = list(done.values())
    print(f"[{method}] run{RUN} todo {len(todo)}/{len(items)}", flush=True)

    def answer(it):
        prompt = P.to_small_model_prompt(it.get("prompt") or "")
        kg = it.get("kg_block") or ""
        if kg and _base_method(method) not in SUMMARY_KG_METHODS:
            missing = _evidence_lines(kg) - _evidence_lines(prompt)
            assert not missing, f"BUG: kg_block evidence missing from prompt for {it['uid']}"
        raw = call_llm(prompt, model=MODEL)
        pred = extract_letter(raw, OPTS.get(it["uid"]))
        return {"uid": it["uid"], "gold": it["gold"], "predicted": pred,
                "is_correct": pred == it["gold"], "route": it.get("route"),
                "kg_block": kg, "prompt": prompt, "raw_response": raw}

    def save():
        n = len(results); c = sum(1 for x in results if x["is_correct"])
        none = sum(1 for x in results if x["predicted"] is None)
        m = compute_metrics(results, labels=LABELS) if results else None
        json.dump({"dataset": DATASET, "method": method, "model": MODEL, "run": RUN,
                   "prompt": "small_model_block", "n": n, "n_correct": c,
                   "runs_correct": [c], "mean_correct": c, "std_correct": 0.0,
                   "n_unparseable": none, "runs_unparseable": [none],
                   "mean_unparseable": none,
                   "accuracy": 100 * c / max(1, n), "mean_acc": 100 * c / max(1, n),
                   "std_acc": 0.0, "metrics": m, "metrics_per_run": [m] if m else [],
                   "results": results,
                   "runs": [{"run": RUN, "results": results}]},
                  open(out_path, "w"), indent=1)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, fut in enumerate(as_completed([ex.submit(answer, it) for it in todo])):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  ERR: {repr(e)[:120]}", flush=True)
                continue
            if (i + 1) % 25 == 0:
                save()
                c = sum(1 for x in results if x["is_correct"])
                none = sum(1 for x in results if x["predicted"] is None)
                print(f"  [{method}] {len(results)}/{len(items)} acc={100*c/len(results):.1f}% "
                      f"unparseable={none}", flush=True)
    save()
    c = sum(1 for x in results if x["is_correct"])
    none = sum(1 for x in results if x["predicted"] is None)
    print(f"[{method}] run{RUN} DONE {c}/{len(results)} = {100*c/max(1,len(results)):.2f}% "
          f"(unparseable={none})", flush=True)
