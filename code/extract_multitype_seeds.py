"""Fix D: generate multi-type seeds (disease, finding, complication, lab,
mechanism, drug) from question stem only. Replaces disease-only seed prompt
to support questions asking about non-disease answers (e.g., aud_012 friction
rub, v2_train_3395 stool guaiac, v2_val_4069 microcephaly/stridor).
"""
import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spectrum_textbook import call_llm

PROJECT_ROOT = HERE.parent
CACHE_DIR = PROJECT_ROOT / "cache"

PROMPT = """You are a clinical reasoner generating retrieval hypotheses for a case. Read the patient case below and list 8-12 distinct clinical concepts that could be relevant to ANY plausible question about this case.

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
{"seeds": ["concept 1", "concept 2", ...]}

Patient case:
{question}

JSON:"""


def extract_one(item, model):
    uid = str(item.get("uid", item.get("idx")))
    question = (item.get("question") or "").strip()[:3000]
    prompt = PROMPT.replace("{question}", question)
    try:
        raw = call_llm(prompt, model=model)
    except Exception:
        return uid, None
    if not raw:
        return uid, None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return uid, None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return uid, None
    seeds = parsed.get("seeds")
    if not isinstance(seeds, list):
        return uid, None
    return uid, {"seeds": [str(s).strip() for s in seeds if str(s).strip()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(CACHE_DIR / "benchmark_bench340_v9.json"))
    ap.add_argument("--out",   default=str(CACHE_DIR / "bench340_seeds_multitype.json"))
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    bench = json.load(open(args.bench))
    if args.limit:
        bench = bench[: args.limit]
    out = {}
    # Resume support: load existing
    if Path(args.out).exists():
        try:
            out = json.load(open(args.out))
            print(f"Resume from {len(out)} existing entries")
        except Exception:
            pass

    todo = [it for it in bench if str(it.get("uid")) not in out]
    print(f"To process: {len(todo)} / {len(bench)} items")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(extract_one, it, args.model): it for it in todo}
        for fut in as_completed(futures):
            uid, val = fut.result()
            if val is not None:
                out[uid] = val
            done += 1
            if done % 20 == 0 or done == len(todo):
                elapsed = time.time() - t0
                print(f"[{done}/{len(todo)}] elapsed {elapsed:.1f}s "
                       f"ETA {elapsed/done*(len(todo)-done):.1f}s")
                json.dump(out, open(args.out, "w"), indent=2)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved {len(out)} entries to {args.out}")


if __name__ == "__main__":
    main()
