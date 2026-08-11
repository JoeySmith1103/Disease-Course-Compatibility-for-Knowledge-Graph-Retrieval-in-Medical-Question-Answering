#!/usr/bin/env python3
"""Build the raw 1-hop / 2-hop KG-dump baselines and FREEZE them like every other method.

These are the "no filtering, no ranking" controls: expand the graph from the SAME shared seed
set the walker uses, and dump the concepts in graph-traversal order. No cos, no bc, no scoring —
that is the whole point (they show what selective retrieval buys over dumping neighbours).

Needs Neo4j. Makes NO LLM calls: seeds/durations are read from datasets/<ds>/, so this is fast.
Output: frozen/<ds>/raw_1hop.json and frozen/<ds>/raw_2hop.json, same schema as the other
methods ({uid, gold, route, kg_block, prompt}), so run_reader.py / run_reader_block.py can
point ANY model at them with no extra work.

Usage:  DATASET=329|1273  [RAW_CAP=50]  python3 pipeline/build_raw_hops.py
"""
import json, os, sys, importlib.util
from pathlib import Path

PIPE = Path(__file__).resolve().parent
CODE = PIPE / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "dkr_policy"))
os.environ.setdefault("DKR_MATRIX_PKL", "cache/umls_broad_embeddings_sapbert.pkl")

DS = os.environ.get("DATASET", "1273")
DD = PIPE / f"datasets/{DS}"
CAP = int(os.environ.get("RAW_CAP", "50"))            # concepts RETAINED in the record
# Only PROMPT_K of them are injected into the prompt, so the raw-hop baselines show the SAME
# number of concepts as the walker (top_k=10). Otherwise they would get 5x the context and the
# comparison would be about context size, not about retrieval quality.
PROMPT_K = int(os.environ.get("RAW_PROMPT_K", "10"))

_spec = importlib.util.spec_from_file_location("P", PIPE / "prompts.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)

bench = json.load(open(DD / "benchmark.json"))
seeds_c = json.load(open(DD / "seeds.json"))
sym_c = json.load(open(DD / "symptoms.json"))
dur_c = json.load(open(DD / "durations.json"))
qe_c = json.load(open(DD / "query_entities.json"))


def days_to_phrase(d):
    if isinstance(d, (list, tuple)):
        seen = []
        for x in d:
            p = days_to_phrase(x)
            if p not in seen: seen.append(p)
        return " / ".join(seen) if len(seen) > 1 else seen[0]
    if d < 1:   return f"{int(d*24)} hours"
    if d < 14:  return f"{int(d)} days"
    if d < 60:  return f"{int(d/7)} weeks"
    if d < 365: return f"{int(d/30)} months"
    return f"{int(d/365)} years"


def ob(o):
    return "\n".join(f"  {k}. {v}" for k, v in sorted(o.items()))


def main():
    # Reuse the walker's seed resolution so all methods start from the SAME seeds.
    os.environ.setdefault("BENCH_PATH", str(DD / "benchmark.json"))
    os.environ.setdefault("PD_PATH", str(DD / "durations.json"))
    os.environ.setdefault("SEEDS_PATH", str(DD / "seeds.json"))
    os.environ.setdefault("SYM_PATH", str(DD / "symptoms.json"))
    os.environ.setdefault("QE_PATH", str(DD / "query_entities.json"))
    import eval_kg_walker_full as E
    from umls_neo4j import get_driver
    from kg_walker import _is_clinical_concept
    drv = get_driver()

    def expand(session, cuis, seen, cap):
        """One hop out from `cuis`, in graph-traversal order. No ranking, no scoring."""
        out = []
        for c in cuis:
            rows = session.run(
                "MATCH (a:Concept {CUI:$c})-[r]-(b:Concept) WHERE r.RELA IS NOT NULL "
                "RETURN b.CUI AS cui, b.name AS name LIMIT 200", c=c).data()
            for x in rows:
                cu, nm = x["cui"], x["name"]
                if not cu or cu in seen or not nm or not _is_clinical_concept(nm):
                    continue
                seen.add(cu); out.append((cu, nm))
                if len(out) >= cap: return out
        return out

    frozen = {1: [], 2: []}
    with drv.session() as s:
        for n, item in enumerate(bench, 1):
            uid = item["uid"]
            seed_dicts = E._collect_all_seeds(uid, item, seeds_c, sym_c, max_total=14)
            seed_cuis = [d["cui"] for d in seed_dicts]
            d = (dur_c.get(uid) or {}).get("days")
            dur_str = days_to_phrase(d) if isinstance(d, (int, float)) and d > 0 else P.NO_DURATION_STR

            seen = set(seed_cuis)
            hop1 = expand(s, seed_cuis, seen, CAP)
            # 2-hop = 1-hop frontier expanded once more, appended after it (traversal order)
            hop2 = hop1 + expand(s, [c for c, _ in hop1][:20], seen, max(0, CAP - len(hop1)))

            for hop, nodes in ((1, hop1), (2, hop2)):
                kept = nodes[:CAP]                 # retained for the record / later analysis
                shown = kept[:PROMPT_K]            # what actually goes into the prompt
                kg = "\n".join(f"  {i}. {nm}" for i, (_, nm) in enumerate(shown, 1)) \
                     or "  (no concepts found)"
                pr = P.RAW_KG.format(question=item["question"], options_block=ob(item["options"]),
                                     patient_dur_str=dur_str, kg_block=kg)
                assert kg in pr, f"BUG: kg_block missing from prompt for {uid}"
                frozen[hop].append({"uid": uid, "gold": item["answer"],
                                    "route": f"raw_{hop}hop", "kg_block": kg, "prompt": pr,
                                    "concepts_retrieved": [nm for _, nm in kept]})
            if n % 100 == 0:
                print(f"  [{n}/{len(bench)}]  1hop={len(hop1)} 2hop={len(hop2)}", flush=True)

    (PIPE / f"frozen/{DS}").mkdir(parents=True, exist_ok=True)
    for hop in (1, 2):
        out = PIPE / f"frozen/{DS}/raw_{hop}hop.json"
        json.dump({"method": f"raw_{hop}hop", "dataset": DS, "n": len(frozen[hop]),
                   "items": frozen[hop]}, open(out, "w"), indent=1)
        avg = sum(len(x["kg_block"].splitlines()) for x in frozen[hop]) / max(1, len(frozen[hop]))
        avg_all = sum(len(x["concepts_retrieved"]) for x in frozen[hop]) / max(1, len(frozen[hop]))
        print(f"froze {len(frozen[hop])} -> {out}  (prompt shows {avg:.1f}/question, {avg_all:.1f} retained)")


if __name__ == "__main__":
    main()
