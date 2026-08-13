"""Faithful MedRAG baseline using MedQA's own 18 textbook corpus.

Pipeline (matching MedRAG paper, Xiong et al. 2024):
1. BM25 over 102K chunks from 18 textbooks (MedQA source corpus)
2. Retrieve top-K chunks per question
3. Inject as evidence into LLM prompt
4. Same prompt template as walker (CoT + duration mention)

MedRAG paper reports BM25 + 32 chunks ≈ MedCPT dense (best is RRF). We use
BM25 + K=32 as the primary baseline (their reported default for MedQA).
"""
import json, os, re, sys, time, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
# pipeline/scripts no longer exists; the LLM client lives in pipeline/code. The old path was
# inert — spectrum_textbook happened to resolve from code/ and re-export call_llm — so this was
# working by accident rather than by design.
sys.path.insert(0, str(ROOT/"code"))

CACHE = ROOT/"cache"
TOP_K = int(os.environ.get("MEDRAG_TOP_K", "32"))   # MedRAG default
INDEX_PATH = CACHE/"medqa_textbook_chunks.pkl"

BENCH_PATH = os.environ.get("BENCH_PATH", str(CACHE/"benchmark_bench340_v9.json"))
bench = json.load(open(BENCH_PATH))
pd_c = json.load(open(os.environ.get("MEDRAG_PD", str(CACHE/"bench340_patient_durations.json"))))

PROMPT = """You are an expert medical diagnostician. Solve the multiple-choice medical question using the supplied medical-textbook excerpts as supporting evidence.

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

After your reasoning (under 250 words), state your final answer as a single letter in <a></a> tags. Example: <a>C</a>.
"""

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
    for pat in [r"<a>\s*([A-J])\s*</a>", r"\\boxed\{\s*([A-J])\s*\}",
                 r"\bfinal answer\s*(?:is|:)\s*\**\(?([A-J])\)?\b",
                 r"\bthe answer\s*(?:is|:)\s*\**\(?([A-J])\)?\b",
                 r"\*\*?Answer\*?\*?\s*:?\s*\**([A-J])\b"]:
        m = re.search(pat, raw, re.IGNORECASE)
        if m: return m.group(1).upper()
    return None

def tok(t):
    return [w for w in re.findall(r"[a-z]+", (t or "").lower()) if len(w) >= 3]

# Worker-local index loading (one per process)
_INDEX = {}
def _load_index():
    if _INDEX: return _INDEX["chunks"], _INDEX["bm25"]
    with open(INDEX_PATH, "rb") as f:
        d = pickle.load(f)
    _INDEX["chunks"] = d["chunks"]; _INDEX["bm25"] = d["bm25"]
    return d["chunks"], d["bm25"]

def process_one(item):
    from llm_client import call_llm
    chunks, bm25 = _load_index()
    uid = item["uid"]
    gold = item["answer"]
    opts = item["options"]
    q = item["question"]
    q_tokens = tok(q)
    scores = bm25.get_scores(q_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:TOP_K]
    evidence_lines = []
    for rank, i in enumerate(top_idx, 1):
        c = chunks[i]
        # Trim each chunk slightly to keep total prompt manageable
        snippet = c["text"][:500]
        evidence_lines.append(f"[Excerpt {rank}] ({c['book']})\n{snippet}\n")
    evidence_block = "\n".join(evidence_lines)
    # The 2000-char cut is kept ON PURPOSE, and it is applied to the PROMPT, not only to the stored
    # record. frozen/329/medrag.json was produced this way — its prompt announces "top-32 BM25
    # matches" but carries only the ~4 excerpts that fit in 2000 characters. Sending the full k=32
    # block for MedBullets/MMLU would give those two datasets eight times the evidence under the
    # same method name, so the three would not be comparable. The mismatch between the label and
    # the content is inherited from 329 and is a property of the shipped baseline, not of this run.
    evidence_block = evidence_block[:2000]
    opts_block = "\n".join(f"  {k}. {v}" for k, v in sorted(opts.items()))
    pd_days = (pd_c.get(uid) or {}).get("days")
    prompt = PROMPT.format(question=q, options_block=opts_block,
                           patient_dur_str=days_to_phrase(pd_days),
                           evidence_block=evidence_block, top_k=TOP_K)
    try:
        raw = call_llm(prompt, model=os.environ.get("WALKER_LLM", "gpt-5.4-mini"))
    except Exception as e:
        return {"uid": uid, "gold": gold, "predicted": None, "is_correct": False,
                "route": "llm_error", "kg_block": evidence_block,
                "prompt_full": prompt, "raw_response": f"ERROR: {e}"}
    pred = extract_letter(raw)
    # prompt_full is required, not optional: freeze_baseline.py drops any record without a stored
    # prompt, so the previous output froze to n=0 and frozen/329/medrag.json ended up with an
    # empty kg_block on all 329 items. Storing the same truncated block that went into the prompt
    # is what lets the freeze-time containment check pass — and what makes these files
    # re-renderable, which 329's are not.
    return {"uid": uid, "gold": gold, "predicted": pred,
            "is_correct": pred == gold, "route": f"medrag_textbook_k{TOP_K}",
            "kg_block": evidence_block, "prompt_full": prompt, "raw_response": raw,
            "n_retrieved_chunks": len(top_idx)}

def main():
    N_WORKERS = int(os.environ.get("WALKER_N_WORKERS","8"))   # lower because index is big
    out = {"results": [], "metadata": {"n_total": len(bench),
            "model": os.environ.get("WALKER_LLM","gpt-5.4-mini"),
            "retriever": "BM25", "top_k": TOP_K, "corpus": "MedQA textbooks (18 books, 102K chunks)"}}
    t0 = time.time()
    model_tag = os.environ.get("WALKER_LLM","gpt-5.4-mini").replace("/","_")
    save_path = CACHE/f"{os.environ.get('MEDRAG_OUT_TAG','bench340')}_medrag_textbook_k{TOP_K}__{model_tag}.json"
    completed = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        future_to_uid = {ex.submit(process_one, it): it["uid"] for it in bench}
        for fut in as_completed(future_to_uid):
            try: res = fut.result()
            except Exception as e:
                print(f"  ERR: {e}", flush=True)
                continue
            out["results"].append(res)
            completed += 1
            if completed % 10 == 0 or completed <= 5:
                n_ok = sum(1 for r in out["results"] if r.get("is_correct"))
                elapsed = time.time() - t0
                eta = elapsed/completed*(len(bench)-completed)/60
                print(f"[{completed}/{len(bench)}] → {res.get('predicted')} acc={n_ok}/{completed}={100*n_ok/completed:.1f}%  ETA {eta:.1f}min", flush=True)
    n_cor = sum(1 for r in out["results"] if r.get("is_correct"))
    out["metadata"]["n_correct"] = n_cor
    out["metadata"]["accuracy"] = 100*n_cor/max(1,len(out["results"]))
    json.dump(out, open(save_path, "w"), indent=2)
    print(f"\nSaved → {save_path}  acc={out['metadata']['accuracy']:.2f}%  elapsed={(time.time()-t0)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
