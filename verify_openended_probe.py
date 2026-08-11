#!/usr/bin/env python3
"""Second arm of the verification: does the MCQ format hide a real duration effect?

verify_duration_critical.py can only observe a flip that the option set permits. In MedQA the
distractors are rarely the acute/chronic counterpart of the gold, so a question can be genuinely
duration-dependent and still be scored not_duration_critical simply because there was nothing to
flip TO. That failure mode looks identical to "duration does not matter", and the difference
matters: one is a property of medicine, the other a property of the benchmark format.

This probe re-runs the same original/perturbed pair with NO OPTIONS — free-text "most likely
diagnosis" — on the questions the MCQ arm called not_duration_critical:

  open_shift    the free-text diagnosis changes with duration although the MCQ answer did not
                → duration IS load-bearing; the option set was the binding constraint
  open_stable   neither changes → duration genuinely carries no discriminative weight here

A separate judge call decides whether two free-text diagnoses are the same condition, so wording
("SBP" vs "spontaneous bacterial peritonitis") is not mistaken for a shift.

Usage:
  DATASET=329 SAMPLE=80 python3 pipeline/verify_openended_probe.py
Output: pipeline/verification/<ds>_openended_<model>.json  (resumable)
"""
import json, os, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "code"))
from llm_client import call_llm

DATASET = os.environ.get("DATASET", "329")
MODEL   = os.environ.get("MODEL", "gpt-5.4-mini")
SAMPLE  = int(os.environ.get("SAMPLE", "80"))     # 0 = every eligible question
WORKERS = int(os.environ.get("WORKERS", "10"))
VDIR = PIPE / "verification"
OUT_FILE = VDIR / f"{DATASET}_openended_{MODEL.replace('/','_')}.json"

SRC = json.load(open(next(VDIR.glob(f"{DATASET}_duration_critical_*.json"))))
# only questions that reached the perturbed vote and did NOT flip are informative here
POOL = [r for r in SRC["results"]
        if r["verdict"] == "not_duration_critical" and r.get("modified_question")]
POOL.sort(key=lambda r: r["uid"])                  # deterministic, no Math.random equivalent
if SAMPLE: POOL = POOL[:SAMPLE]

OPEN = """You are an expert clinician. Read the vignette and state the single most likely diagnosis.

{question}

Answer with the diagnosis name only — no explanation, no differential, no more than 8 words."""

SAME = """Are these two clinical diagnoses the same condition (allowing for synonyms, abbreviations,
and differences in specificity)?

A: {a}
B: {b}

Answer with one word: SAME or DIFFERENT."""

# A name that changed is not yet evidence of anything: "anal fissure" -> "chronic anal fissure" is
# one entity restated, while "acute mesenteric ischemia" -> "chronic mesenteric ischemia" is two
# entities with different management despite sharing a stem. Nomenclature cannot separate those,
# so the second judge asks the clinical question instead.
DISTINCT = """Two clinicians proposed these diagnoses for the same patient:

A: {a}
B: {b}

Would A and B lead to DIFFERENT immediate clinical management (different urgency, different first
investigation, or different first treatment)?

Answer with exactly one word:
  DIFFERENT   — genuinely different entities requiring different management
  SAME        — the same entity restated, or a difference in wording, specificity or stage that
                does not change management
  INVALID     — at least one of them is not a diagnosis (e.g. a test, a drug, an organism vector)"""


def probe(rec):
    q_orig = next(b["question"] for b in BENCH if b["uid"] == rec["uid"])
    a = (call_llm(OPEN.format(question=q_orig), model=MODEL) or "").strip().split("\n")[0]
    b = (call_llm(OPEN.format(question=rec["modified_question"]), model=MODEL) or "").strip().split("\n")[0]
    same = None
    if a and b:
        v = (call_llm(SAME.format(a=a, b=b), model=MODEL) or "").upper()
        same = "SAME" in v and "DIFFERENT" not in v
    return {"uid": rec["uid"],
            "duration_change": f"{rec['perturb']['original_duration']} -> {rec['perturb']['new_duration']}",
            "open_dx_original": a, "open_dx_perturbed": b,
            "verdict": ("open_stable" if same else "open_shift") if same is not None else "probe_fail"}


BENCH = json.load(open(PIPE / f"datasets/{DATASET}/benchmark.json"))

if __name__ == "__main__":
    done = {}
    if OUT_FILE.exists():
        try: done = {r["uid"]: r for r in json.load(open(OUT_FILE))["results"]}
        except Exception: done = {}
    todo = [r for r in POOL if r["uid"] not in done]
    print(f"[open-probe] {DATASET} · {len(todo)} to go ({len(done)} cached) "
          f"from {len(POOL)} not_duration_critical questions", flush=True)
    results = list(done.values())

    def save():
        json.dump({"dataset": DATASET, "model": MODEL, "n": len(results),
                   "verdicts": dict(Counter(r["verdict"] for r in results)),
                   "results": results}, open(OUT_FILE, "w"), indent=1)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(probe, r): r["uid"] for r in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try: results.append(f.result())
            except Exception as e: print(f"  ERR {futs[f]}: {repr(e)[:80]}", flush=True); continue
            if i % 20 == 0: save(); print(f"  [{len(results)}/{len(POOL)}] "
                                          f"{dict(Counter(r['verdict'] for r in results))}", flush=True)
    save()

    # second pass: keep only the shifts that change management (resumable on the new field)
    need = [r for r in results if r["verdict"] == "open_shift" and "management" not in r]
    if need:
        print(f"\n[re-judge] {len(need)} shifts -> does the change alter management?", flush=True)
        def rj(r):
            v = (call_llm(DISTINCT.format(a=r["open_dx_original"], b=r["open_dx_perturbed"]),
                          model=MODEL) or "").upper()
            r["management"] = ("INVALID" if "INVALID" in v else
                               "DIFFERENT" if "DIFFERENT" in v else "SAME")
            return r
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(rj, need))
        save()

    c = Counter(r["verdict"] for r in results)
    n = c["open_shift"] + c["open_stable"]
    m = Counter(r.get("management") for r in results if r["verdict"] == "open_shift")
    print(f"\n[open-probe] {DATASET}: {dict(c)}")
    if n:
        print(f"  MCQ scored these not_duration_critical; the free-text diagnosis still moved in "
              f"{c['open_shift']}/{n} = {100*c['open_shift']/n:.1f}%")
        print(f"  of those shifts: {dict(m)}")
        strict = m["DIFFERENT"]
        valid = n - m["INVALID"]
        print(f"  → after discarding restatements and non-diagnoses: {strict}/{valid} = "
              f"{100*strict/valid:.1f}% carry a duration effect the OPTION SET hid, not one absent "
              f"from the medicine.")
    print(f"  -> {OUT_FILE}")
