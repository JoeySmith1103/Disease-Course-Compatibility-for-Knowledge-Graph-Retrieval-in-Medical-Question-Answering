#!/usr/bin/env python3
"""Run OUR method end-to-end on ONE held-out training question and dump the full trace,
for use as a one-shot in-context example.

The question is from the temporal-critical v2 TRAIN split — NOT in the 329 or 1273 eval sets.
Every input is generated exactly as the pipeline generates it:
  * seeds     — the real multitype-seed prompt (extract_multitype_seeds.PROMPT)
  * duration  — the real role-tagged LLM extractor (extract_patient_duration_llm)
  * symptoms + query_entities — LLM, matching the shapes the walker consumes
Then eval_kg_walker_full.process_one runs the actual walk (score = cos + λ·bc − μ·hop) and the
reader answers, so the dumped kg_block / prompt / answer are exactly what the method produces.

Usage:  python3 pipeline/build_oneshot_example.py
Output: pipeline/oneshot/example.json  (+ human-readable example.md)
"""
import json, os, re, sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
CODE = PIPE / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "dkr_policy"))
os.environ.setdefault("DKR_MATRIX_PKL", "cache/umls_broad_embeddings_sapbert.pkl")

from llm_client import call_llm
MODEL = os.environ.get("ONESHOT_LLM", "gpt-5.4-mini")
OUT = PIPE / "oneshot"; OUT.mkdir(exist_ok=True)

# ── the chosen held-out training question (precocious puberty → Granulosa cell tumor) ──
# temporal_critical_v2/train, not in 329 or 1273. Diagnosis-type MCQ (all disease options) with a
# numeric chief-complaint duration ("past 2 months") so the duration term (bc) is genuinely active.
UID = "oneshot_precocious"
Q = ("A 5-year-old girl is brought to the clinic by her mother for excessive hair growth. Her "
     "mother reports that for the past 2 months she has noticed hair at the axillary and pubic "
     "areas. She denies any family history of precocious puberty and reports that her daughter has "
     "been relatively healthy with an uncomplicated birth history. She denies any recent illnesses, "
     "weight change, fever, vaginal bleeding, pain, or medication use. Physical examination "
     "demonstrates Tanner stage 4 development. A pelvic ultrasound shows an ovarian mass. Laboratory "
     "studies demonstrates an elevated level of estrogen. What is the most likely diagnosis?")
OPTIONS = {"A": "Congenital adrenal hyperplasia", "B": "Granulosa cell tumor",
           "C": "Idiopathic precocious puberty", "D": "McCune-Albright syndrome",
           "E": "Sertoli-Leydig tumor"}
GOLD = "B"

# ── seeds: the real multitype-seed prompt, verbatim ──
from extract_multitype_seeds import PROMPT as SEED_PROMPT

def gen_seeds():
    raw = call_llm(SEED_PROMPT.replace("{question}", Q), model=MODEL)
    m = re.search(r"\{[\s\S]*\}", raw)
    return [str(s).strip() for s in json.loads(m.group(0))["seeds"] if str(s).strip()]

# ── duration: the real role-tagged LLM extractor, verbatim ──
from extract_patient_duration_llm import extract_patient_duration_llm

# ── symptoms + query_entities: LLM, matching the shapes the walker consumes ──
QE_PROMPT = """You are a clinical information extractor. From the vignette, extract:
- symptoms_signs: presenting symptoms, signs, and abnormal findings (short phrases)
- diseases_mentioned: any diseases/conditions explicitly named in the case (comorbidities, prior dx)
- procedures: procedures or interventions mentioned
- drugs: drugs/medications mentioned
- lab_findings: lab tests or lab results mentioned
Use ONLY the case text. Output strict JSON with exactly those five keys, each a list of strings.
Vignette:
\"\"\"{q}\"\"\"
JSON:"""

def gen_query_entities():
    raw = call_llm(QE_PROMPT.replace("{q}", Q), model=MODEL)
    m = re.search(r"\{[\s\S]*\}", raw)
    d = json.loads(m.group(0))
    for k in ("symptoms_signs", "diseases_mentioned", "procedures", "drugs", "lab_findings"):
        d.setdefault(k, [])
        d[k] = [str(x).strip() for x in d[k] if str(x).strip()]
    return d


def main():
    print("[1/4] generating seeds (multitype prompt) ...", flush=True)
    seeds = gen_seeds()
    print("      seeds:", seeds)

    print("[2/4] extracting patient duration (role-tagged LLM) ...", flush=True)
    dur = extract_patient_duration_llm(Q, model=MODEL)
    print(f"      chief-complaint days={dur.get('days')} role={dur.get('role')} span={dur.get('span')!r}")

    print("[3/4] extracting symptoms + query entities ...", flush=True)
    qe = gen_query_entities()
    symptoms = qe["symptoms_signs"]
    print("      symptoms:", symptoms)

    # write mini dataset files the walker reads, then run the REAL walker
    dd = OUT / "dataset"; dd.mkdir(exist_ok=True)
    json.dump([{"uid": UID, "question": Q, "options": OPTIONS, "answer": GOLD}], open(dd / "benchmark.json", "w"), indent=1)
    json.dump({UID: {"seeds": seeds}}, open(dd / "seeds.json", "w"), indent=1)
    json.dump({UID: {"symptoms": symptoms}}, open(dd / "symptoms.json", "w"), indent=1)
    json.dump({UID: dur}, open(dd / "durations.json", "w"), indent=1)
    json.dump({UID: qe}, open(dd / "query_entities.json", "w"), indent=1)

    print("[4/4] running the walker (cos + 0.3·bc − 0.08·hop) + reader ...", flush=True)
    # start fresh — eval_kg_walker_full resumes from this checkpoint if present (would keep a
    # previous question's result under a stale uid).
    (PIPE / f"cache/oneshot__{MODEL.replace('/','_')}.json").unlink(missing_ok=True)
    env = dict(os.environ,
        BENCH_PATH=str(dd / "benchmark.json"), PD_PATH=str(dd / "durations.json"),
        SEEDS_PATH=str(dd / "seeds.json"), SYM_PATH=str(dd / "symptoms.json"),
        QE_PATH=str(dd / "query_entities.json"),
        WALKER_LLM=MODEL, WALKER_MULTI_AXIS="1", WALKER_RETRIEVAL_ONLY="0",
        WALKER_N_WORKERS="1", WALKER_OUT_TAG="oneshot",
        # one question: keep it fast — bc from the cache, no per-candidate on-demand LLM calls
        WALKER_BC_CACHE_ONLY="1")
    import subprocess
    subprocess.run([sys.executable, str(CODE / "eval_kg_walker_full.py")], env=env,
                   check=True, cwd=str(PIPE))
    # eval_kg_walker_full writes to pipeline/cache (its own CACHE), so read from there
    rec = json.load(open(PIPE / f"cache/oneshot__{MODEL.replace('/','_')}.json"))
    r = rec["results"][0] if "results" in rec else rec[0]

    out = {
        "uid": UID, "split": "temporal_critical_v2/train (held-out; not in 329 or 1273)",
        "question": Q, "options": OPTIONS, "gold": GOLD,
        "generated_inputs": {
            "seeds": seeds, "symptoms": symptoms, "query_entities": qe,
            "patient_duration": {"days": dur.get("days"), "role": dur.get("role"),
                                 "span": dur.get("span"), "spans": dur.get("spans")},
        },
        "walker_output": {
            "route": r.get("route"), "patient_days": r.get("patient_days"),
            "n_walker_candidates": r.get("n_walker_candidates"),
            "kg_block": r.get("kg_block"),
        },
        "reader_prompt": r.get("prompt") if "prompt" in r else None,
        "predicted": r.get("predicted"), "is_correct": r.get("predicted") == GOLD,
        "reader_reasoning": r.get("raw_response"),
    }
    json.dump(out, open(OUT / "example.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT/'example.json'}")
    print(f"route={out['walker_output']['route']}  predicted={out['predicted']}  gold={GOLD}  "
          f"correct={out['is_correct']}")


if __name__ == "__main__":
    main()
