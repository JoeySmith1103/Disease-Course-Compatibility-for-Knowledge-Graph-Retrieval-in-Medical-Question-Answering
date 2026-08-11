#!/usr/bin/env python3
"""Answer step for SMALL/MEDIUM open models (Qwen etc.) using the BLOCK-structured prompt.

Same contract as run_reader.py — reads FROZEN per-question prompts' kg_block from
frozen/<dataset>/<method>.json and does no retrieval — but re-wraps each question in
prompts.BLOCK_KG / BLOCK_NOKG, which end with a fixed `ANSWER: X` line. Small models
often fail the <a>X</a> convention used by run_reader.py, producing unparseable output.

Resumable + incremental: re-running skips uids already saved, so a rate-limit stall or
crash can be resumed rather than restarted. This matters on Together AI, which applies a
*dynamic* rate limit that throttles bursty workloads (keep WORKERS modest, ~4-6).

Usage:
  DATASET=329|1273  MODEL=together:Qwen/Qwen3.5-9B  RUN=2  WORKERS=6 \
  METHODS=vanilla,cot,medrag,tog,hykge,walker  python3 pipeline/run_reader_block.py
Output: pipeline/results/<dataset>_<method>_<modeltag>_block_run<RUN>.json
"""
import json, os, re, sys, importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "code"))
sys.path.insert(0, str(PIPE / "code" / "dkr_policy"))
from llm_client import call_llm

spec = importlib.util.spec_from_file_location("P", PIPE / "prompts.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

DATASET = os.environ.get("DATASET", "1273")
MODEL   = os.environ.get("MODEL", "together:Qwen/Qwen3.5-9B")
RUN     = os.environ.get("RUN", "1")
WORKERS = int(os.environ.get("WORKERS", "6"))
METHODS = os.environ.get("METHODS", "vanilla,cot,medrag,tog,hykge,walker").split(",")

MODEL_TAG = MODEL.split("/")[-1].replace(":", "_")
RES = PIPE / "results"; RES.mkdir(exist_ok=True)


def extract_letter(raw):
    """ReinRAG output spec yields a '[Answer]' section; the rest are fallbacks."""
    for p in [r"\[Answer\]\s*:?\s*\**([A-J])\b", r"ANSWER:\s*\**([A-J])\b",
              r"<a>\s*([A-J])\s*</a>", r"\bthe answer is\s*:?\s*\**([A-J])\b",
              r"\bfinal answer\s*:?\s*\**([A-J])\b", r"\bAnswer\s*:?\s*\**([A-J])\b"]:
        m = re.search(p, raw or "", re.I)
        if m:
            return m.group(1).upper()
    return None


# Prompt surgery lives in prompts.py (P.to_small_model_prompt) — see that file.



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
        # Send the frozen, method-correct ReinRAG prompt VERBATIM (walker→duration+bc legend,
        # medrag→textbook, etc.) and only override the answer format for robust extraction.
        # Small-model input then differs from the big-model run ONLY in that final instruction.
        prompt = P.to_small_model_prompt(it.get("prompt") or "")
        kg = it.get("kg_block") or ""
        # Canary: a silent kg_block drop once wiped the evidence out of every prompt. Persist
        # kg_block per-uid and assert it actually reached the prompt, so it is auditable BEFORE
        # anyone looks at accuracy.
        assert (not kg) or (kg in prompt), f"BUG: kg_block missing from prompt for {it['uid']}"
        raw = call_llm(prompt, model=MODEL)
        pred = extract_letter(raw)
        # ALWAYS persist the exact prompt sent and the raw model output, not just the verdict.
        return {"uid": it["uid"], "gold": it["gold"], "predicted": pred,
                "is_correct": pred == it["gold"], "route": it.get("route"),
                "kg_block": kg, "prompt": prompt, "raw_response": raw}

    def save():
        n = len(results); c = sum(1 for x in results if x["is_correct"])
        none = sum(1 for x in results if x["predicted"] is None)
        json.dump({"dataset": DATASET, "method": method, "model": MODEL, "run": RUN,
                   "prompt": "block", "n": n, "n_correct": c, "n_unparseable": none,
                   "accuracy": 100 * c / max(1, n), "results": results},
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
    # A high unparseable count usually means rate-limit/timeout failures, NOT model error.
    print(f"[{method}] run{RUN} DONE {c}/{len(results)} = {100*c/max(1,len(results)):.2f}% "
          f"(unparseable={none})", flush=True)
