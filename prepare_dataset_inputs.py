#!/usr/bin/env python3
"""Generate the three LLM-derived retrieval inputs a new dataset needs before build_kg.py can run.

build_kg.py wires four files into the walker: benchmark.json (written by build_new_dataset.py),
durations.json (extract_patient_duration_llm.py), and these three:

  seeds.json           LLM differential-diagnosis hypotheses  -> walker seed CUIs
  symptoms.json        presenting symptoms/signs              -> the SapBERT query embedding
  query_entities.json  entities split by role                 -> extra multi-role seeds

The prompts are carried over verbatim from _archive/scripts/{extract_multitype_seeds,
extract_symptoms,extract_query_entities}.py so a new dataset is prepared exactly the way 329 and
1273 were; only the import of call_llm is updated (spectrum_textbook -> llm_client). intents.json
is NOT generated — build_kg.py stopped reading it when intent-aware retrieval was removed.

Each file is written incrementally and keyed by uid, so a re-run resumes rather than restarting.

Usage:
  DATASET=medbullets MODEL=gpt-5.4-mini WORKERS=8 python3 pipeline/prepare_dataset_inputs.py
  DATASET=mmlu ONLY=symptoms python3 pipeline/prepare_dataset_inputs.py     # regenerate just one
"""
import json, os, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "code"))
from llm_client import call_llm

DATASET = os.environ.get("DATASET", "medbullets")
MODEL   = os.environ.get("MODEL", "gpt-5.4-mini")
WORKERS = int(os.environ.get("WORKERS", "8"))
ONLY    = os.environ.get("ONLY", "")          # seeds | symptoms | query_entities
DD = PIPE / f"datasets/{DATASET}"

SEEDS_PROMPT = """You are a clinical reasoner generating retrieval hypotheses for a case. Read the patient case below and list 8-12 distinct clinical concepts that could be relevant to ANY plausible question about this case.

Cover diverse concept types as appropriate:
- Diseases / syndromes (most common)
- Findings / signs / lab abnormalities expected in plausible diagnoses
- Complications / sequelae
- Mechanisms / pathophysiology terms
- Drugs / procedures
- Lab tests / imaging modalities

Constraints:
- Use ONLY the case description; do NOT consider answer options.
- Each entry is a canonical concept name (e.g., "Crohn disease", "Friction rub", "Papillary muscle rupture", "Hashimoto thyroiditis", "Positive stool guaiac test").
- Use specific concept names, not generic categories.
- Include the 4-6 most-likely DIAGNOSES at minimum.
- Then add 2-6 NON-DISEASE concepts (typical findings, classic complications, key lab tests, characteristic drugs) the case might hinge on.

Output strict JSON only:
{{"seeds": ["concept 1", "concept 2", ...]}}

Patient case:
{question}

JSON:"""

SYMPTOMS_PROMPT = """You are a medical entity extractor. From the patient case below, list ONLY the presenting symptoms, signs, and pertinent positive physical findings — the complaints/observations a clinician would use to generate a differential diagnosis. Exclude demographics, family history, social history, lab values, and imaging unless they describe a finding.

Output as a compact JSON object with exactly two keys:
  "symptoms": list of short noun phrases (3-10 items typical)
  "duration_summary": brief description of the timing if mentioned (or null)

Patient case:
{question}

JSON:"""

QE_PROMPT = """Extract clinical entities from the medical question below.
Output STRICT JSON with keys, lists of short noun-phrase entities (each <=6 words):
  - symptoms_signs:        present-illness symptoms/signs (the chief complaint and exam findings).
  - diseases_mentioned:    diseases the patient has (past medical history, comorbidities). Exclude options.
  - procedures:            procedures or surgeries the patient had/is having.
  - drugs:                 medications the patient is on.
  - lab_findings:          lab results, imaging findings (specific abnormal values).

EXCLUDE: anything in the answer options. Only entities from the case description (vignette).

Question:
{question}

Output JSON only, no commentary:"""

JOBS = {
    "seeds":          (SEEDS_PROMPT,     "seeds.json",          {"seeds": []}),
    "symptoms":       (SYMPTOMS_PROMPT,  "symptoms.json",       {"symptoms": [], "duration_summary": None}),
    "query_entities": (QE_PROMPT,        "query_entities.json", {"symptoms_signs": [], "diseases_mentioned": [],
                                                                 "procedures": [], "drugs": [], "lab_findings": []}),
}


def extract(prompt, question, empty):
    raw = call_llm(prompt.format(question=question[:3000]), model=MODEL) or ""
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m: return dict(empty)
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return dict(empty)
    # keep only the expected keys so a chatty model cannot inject extras downstream
    return {k: d.get(k, v) for k, v in empty.items()}


def run(job):
    prompt, fname, empty = JOBS[job]
    bench = json.load(open(DD / "benchmark.json"))
    out_path = DD / fname
    out = json.load(open(out_path)) if out_path.exists() else {}
    todo = [b for b in bench if b["uid"] not in out]
    print(f"[{DATASET}/{job}] {len(todo)} to go ({len(out)} cached) of {len(bench)}", flush=True)
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(extract, prompt, b["question"], empty): b["uid"] for b in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                out[futs[f]] = f.result()
            except Exception as e:
                print(f"  ERR {futs[f]}: {repr(e)[:80]}", flush=True); continue
            if i % 25 == 0:
                json.dump(out, open(out_path, "w"), indent=1)
                print(f"  [{len(out)}/{len(bench)}]", flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
    n = sum(1 for v in out.values() if any(v.get(k) for k in empty))
    print(f"[{DATASET}/{job}] wrote {len(out)} entries ({n} non-empty) -> {out_path}")


if __name__ == "__main__":
    for job in ([ONLY] if ONLY else list(JOBS)):
        run(job)
