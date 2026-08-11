"""Re-run all 6 methods on bench329 with FULL prompt + LLM-output storage.

For retrieval-based methods (raw_1hop, raw_2hop, MedRAG, walker), we reuse the
EXISTING kg_block from the original file (no need to redo retrieval) — only
the LLM call is repeated. This confirms LLM stability AND saves prompts.

For prompt-only methods (vanilla, cot_minimal), we just re-call the LLM.

Output per item:
  uid, gold, predicted, is_correct, raw_response, prompt_full

Output metadata:
  model, method, prompt_template, n_total, n_correct, accuracy, run_timestamp
"""
import json, os, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
from spectrum_textbook import call_llm

CACHE = ROOT/"cache"; RES = ROOT/"results"
# Env overrides so the IDENTICAL script (same prompts/reader/extraction) can run on the full 1273.
_BENCH_PATH = os.environ.get("RERUN_BENCH", str(CACHE/"benchmark_bench329_clean.json"))
_PD_PATH    = os.environ.get("RERUN_PD",    str(CACHE/"bench340_patient_durations.json"))
_OUT_TAG    = os.environ.get("RERUN_TAG",   "rerun329")
BENCH = json.load(open(_BENCH_PATH))
PD = json.load(open(_PD_PATH))


# ============ PROMPT TEMPLATES (verbatim from source scripts) ============

PROMPT_VANILLA = """Solve the multiple-choice medical question. Provide your final answer as a single letter in <a></a> tags.

Question:
{question}

Options:
{options_block}

<a>?</a>"""

PROMPT_COT_MINIMAL = (
    "Solve the multiple-choice medical question. Think step by step "
    "before giving your final answer. State your final answer as a "
    "single letter in <a></a> tags. Example: <a>C</a>\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options_block}"
)

PROMPT_RAW = """You are an expert medical diagnostician. Solve the multiple-choice medical question by reasoning step by step.

Question:
{question}

Options:
{options_block}

Supplementary evidence from a clinical knowledge graph (use as ONE input among many; not authoritative):
- Patient symptom duration: {patient_dur_str}
- Top candidate concepts retrieved from SNOMED via raw graph expansion from seeds (NO filtering, NO ranking — listed in graph-traversal order, may contain noise).

{kg_block}

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. Rank the candidates and select the most likely answer.

After your reasoning (under 200 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>."""

PROMPT_MEDRAG = """You are an expert medical diagnostician. Solve the multiple-choice medical question using the supplied medical-textbook excerpts as supporting evidence.

Question:
{question}

Options:
{options_block}

Retrieved textbook excerpts (top-{top_k} BM25 matches from the MedQA source corpus — Harrison, First Aid, Pathoma, etc.):
- Patient symptom duration: {patient_dur_str}

{evidence_block}

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. Cross-reference the retrieved textbook excerpts to support or rule out candidates.
4. Rank the candidates and select the most likely answer.

After your reasoning (under 250 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>."""

PROMPT_WALKER = """You are an expert medical diagnostician. Solve the multiple-choice medical question by reasoning step by step.

Question:
{question}

Options:
{options_block}

Supplementary evidence from a clinical knowledge graph (use as ONE input among many; not authoritative):
- Patient symptom duration: {patient_dur_str}
- Top candidate concepts retrieved from SNOMED via graph expansion from seeds (DDx hypotheses + case findings from the vignette only; option text was NOT used as seeds).

{kg_block}

How to read each entry:
  `N. [role] name (score=S: cos=C+bc=B)`
     path: origin_seed -[rela1]-> mediator -[rela2]-> candidate
  - `score = cos + 0.3·bc - 0.08·hop`: walker's combined ranking signal.
  - `cos`: semantic similarity between the case symptoms and this concept (0..1).
  - `bc` (dur-compat): Bhattacharyya overlap between patient duration and the disease's typical clinical course (0..1; 0 if non-disease).
  - `role`: UMLS TUI-derived category - [disease] / [finding] / [organism] / [procedure] / [anatomy] etc.
  - `path`: shows how the candidate was reached. `seed:` prefix means it was a starting hypothesis (hop=0, high-confidence entry); otherwise the chain shows which SNOMED relations were traversed (e.g., `inverse_isa` = hierarchical subtype, `has_finding_site` = anatomic locus, `due_to` = causal, `associated_finding_of` = clinically co-occurring).
  - Use as a cross-reference; do NOT let it override clinical judgment.

Think step by step:
1. Identify the key clinical findings, demographics, and duration in the case.
2. Enumerate your differential diagnosis (3-5 candidates) based on the presentation.
3. For each candidate, evaluate compatibility with the patient's symptom duration and other clinical features. Use the KG paths to verify which candidates have well-supported retrieved-evidence chains; high cos but low bc may indicate semantic match without duration fit.
4. Rank the candidates and select the most likely answer.

After your reasoning (under 300 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>."""


# ============ HELPERS ============

def days_to_phrase(d):
    if d is None: return "unspecified"
    if isinstance(d, (list, tuple)): d = d[0] if d else None
    if d is None: return "unspecified"
    if d < 1:    return f"{int(d*24)} hours"
    if d < 14:   return f"{int(d)} days"
    if d < 60:   return f"{int(d/7)} weeks"
    if d < 365:  return f"{int(d/30)} months"
    return f"{int(d/365)} years"


def extract_letter(raw):
    if not raw: return None
    for pat in [
        r"<a>\s*([A-Z])\s*</a>",
        r"\\boxed\{\s*([A-Z])\s*\}",
        r"\bfinal answer\s*(?:is|:)\s*\**\(?([A-Z])\)?\b",
        r"\bthe answer\s*(?:is|:)\s*\**\(?([A-Z])\)?\b",
        r"\banswer\s*(?:is|:)\s*\**\(?([A-Z])\)?\b",
    ]:
        m = re.search(pat, raw, re.IGNORECASE)
        if m: return m.group(1).upper()
    tail = raw[-200:]
    m = re.findall(r"\b([A-Z])\b", tail)
    if m: return m[-1]
    return None


# ============ PER-METHOD PROMPT BUILDING ============

def load_existing_kg(method):
    """Load kg_block from existing files for retrieval methods."""
    mapping = {
        "raw_1hop":   CACHE/"bench340_raw_1hop__gpt-5.4-mini.json",
        "raw_2hop":   CACHE/"bench340_raw_2hop__gpt-5.4-mini.json",
        "MedRAG":     CACHE/"bench340_medrag_textbook_k32__gpt-5.4-mini.json",
        "walker_mtS": CACHE/"bench340_walker_fixABCmtS__gpt-5.4-mini.json",
    }
    if method not in mapping: return {}
    obj = json.load(open(mapping[method]))
    return {r["uid"]: r.get("kg_block","") for r in obj["results"]}


def build_prompt(method, item, kg_blocks):
    opts = "\n".join(f"  {k}. {v}" for k, v in sorted(item["options"].items()))
    q = item["question"]
    uid = item["uid"]
    pd_days = PD.get(uid, {}).get("days")
    pd_str = days_to_phrase(pd_days)

    if method == "vanilla":
        return PROMPT_VANILLA.format(question=q, options_block=opts)
    if method == "cot_minimal":
        return PROMPT_COT_MINIMAL.format(question=q, options_block=opts)
    if method == "raw_1hop" or method == "raw_2hop":
        return PROMPT_RAW.format(question=q, options_block=opts,
                                 patient_dur_str=pd_str,
                                 kg_block=kg_blocks.get(uid, "(no KG)"))
    if method == "MedRAG":
        return PROMPT_MEDRAG.format(question=q, options_block=opts,
                                    top_k=32, patient_dur_str=pd_str,
                                    evidence_block=kg_blocks.get(uid, "(no excerpts)"))
    if method == "walker_mtS":
        return PROMPT_WALKER.format(question=q, options_block=opts,
                                    patient_dur_str=pd_str,
                                    kg_block=kg_blocks.get(uid, "(no KG)"))
    raise ValueError(f"unknown method {method}")


def run_one(method, item, kg_blocks, model):
    prompt = build_prompt(method, item, kg_blocks)
    try:
        raw = call_llm(prompt, model=model)
    except Exception as e:
        raw = f"[ERROR] {e}"
    pred = extract_letter(raw)
    gold = item["answer"]
    return {
        "uid": item["uid"], "gold": gold, "predicted": pred,
        "is_correct": (pred == gold), "raw_response": raw,
        "prompt_full": prompt,
    }


def main():
    method = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-5.4-mini"

    valid = ["vanilla", "cot_minimal", "raw_1hop", "raw_2hop", "MedRAG", "walker_mtS"]
    if method not in valid:
        print(f"Unknown method {method}. Valid: {valid}")
        sys.exit(1)

    model_safe = model.replace("/", "_").replace(":", "_")
    out_path = CACHE / f"{_OUT_TAG}_{method}__{model_safe}.json"
    kg_blocks = load_existing_kg(method)
    print(f"  Loaded kg_blocks for {len(kg_blocks)} uids" if kg_blocks else "  No KG required")

    # Resume
    existing = {}
    if out_path.exists():
        try:
            d = json.load(open(out_path))
            for r in d.get("results", []):
                existing[r["uid"]] = r
            print(f"  Resume: {len(existing)} done")
        except: pass

    todo = [it for it in BENCH if it["uid"] not in existing]
    results = list(existing.values())
    print(f"  Method={method}  model={model}  todo={len(todo)}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, method, it, kg_blocks, model): it for it in todo}
        done = 0
        for fut in as_completed(futs):
            try: r = fut.result()
            except Exception as e:
                it = futs[fut]
                r = {"uid": it["uid"], "gold": it["answer"], "predicted": None,
                     "is_correct": False, "raw_response": f"[ERROR] {e}",
                     "prompt_full": ""}
            results.append(r)
            done += 1
            if done % 50 == 0:
                n_c = sum(1 for r_ in results if r_.get("is_correct"))
                print(f"  [{len(results)}/{len(BENCH)}] acc={100*n_c/len(results):.2f}%  elapsed={(time.time()-t0)/60:.1f}min")
                # Checkpoint
                _save(out_path, method, model, results, t0)

    n_c = sum(1 for r in results if r.get("is_correct"))
    _save(out_path, method, model, results, t0)
    print(f"\n=== {method} | {model} ===")
    print(f"  {n_c}/{len(results)} = {100*n_c/len(results):.2f}%")
    print(f"  Saved {out_path}")


def _save(out_path, method, model, results, t0):
    templates = {
        "vanilla":     PROMPT_VANILLA,
        "cot_minimal": PROMPT_COT_MINIMAL,
        "raw_1hop":    PROMPT_RAW,
        "raw_2hop":    PROMPT_RAW,
        "MedRAG":      PROMPT_MEDRAG,
        "walker_mtS":  PROMPT_WALKER,
    }
    n_c = sum(1 for r in results if r.get("is_correct"))
    json.dump({
        "results": results,
        "metadata": {
            "model": model, "method": method,
            "dataset": os.path.basename(_BENCH_PATH),
            "n_total": len(results), "n_correct": n_c,
            "accuracy": 100*n_c/len(results) if results else 0,
            "elapsed_sec": time.time() - t0,
            "prompt_template": templates[method],
            "reuses_kg_from": {
                "raw_1hop":   "cache/bench340_raw_1hop__gpt-5.4-mini.json",
                "raw_2hop":   "cache/bench340_raw_2hop__gpt-5.4-mini.json",
                "MedRAG":     "cache/bench340_medrag_textbook_k32__gpt-5.4-mini.json",
                "walker_mtS": "cache/bench340_walker_fixABCmtS__gpt-5.4-mini.json",
            }.get(method, "n/a"),
        },
    }, open(out_path, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
