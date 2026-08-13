#!/usr/bin/env python3
"""Collect every method's FULL retrieved candidate pool into pool/<dataset>/<method>.jsonl.

Why a pool and not just the frozen prompt: frozen/<ds>/<method>.json holds the top-10 that reached
the reader. Any question about a different top-K, a different λ/μ, or a different utility formula
is unanswerable from it — the discarded candidates are exactly the evidence needed. Storing the
pool with its RAW components turns a parameter sweep into a re-ranking of this file instead of
another walk over Neo4j.

WHAT EACH PARAMETER COSTS, once a pool exists:

  free (re-rank the stored pool)     top_k, λ (bc weight), μ (hop penalty), the utility formula
                                     itself, the role quota, any post-hoc filter
  needs a fresh walk                 τ (min_score), max_hops, neighbor_limit, seeds, the query
                                     embedding — all four gate EXPANSION, so lowering them would
                                     reach nodes that were never scored and cannot be in here

That split is recorded per record in `params.replayable` / `params.needs_rerun`, so an analysis can
check rather than assume.

SOURCES — three of the five methods already persisted their pool, so only the walker re-runs:
  walker, walker_interval   re-walk with WALKER_POOL_DUMP=1     (Neo4j; no LLM if durations cached)
  raw_1hop, raw_2hop        frozen `concepts_retrieved` (50)    (already on disk)
  tog, hykge                cache `chains_full`                 (already on disk)

Usage:
  python3 pipeline/build_pools.py                      # all datasets, from-disk methods only
  python3 pipeline/build_pools.py --walk 329 mmlu      # also re-walk the walker for these
"""
import json, os, subprocess, sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
MODEL = os.environ.get("WALKER_LLM", "gpt-5.4-mini")
DATASETS = ["329", "medbullets", "mmlu"]
WALK = "--walk" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if args:
    DATASETS = args


def _fmt_top10(cands, top_k=10):
    """format_kg_block's exact output, rebuilt from a pool record."""
    dis = [c for c in cands if c.get("role") == "disease"]
    oth = [c for c in cands if c.get("role") != "disease"]
    picked = dis[:7] + oth[:3]
    if len(picked) < top_k:
        seen = {c["cui"] for c in picked}
        for c in cands:
            if c["cui"] in seen: continue
            picked.append(c)
            if len(picked) >= top_k: break
    picked.sort(key=lambda c: -c["score"])
    return "\n".join(f"  {i}. [{c['role']}] {c['name'][:60]} (score={c['score']:.2f}: "
                     f"cos={c['cos']:.2f}+bc={c['bc']:.2f})"
                     for i, c in enumerate(picked[:top_k], 1))


def check_against_frozen(ds, method, records):
    """Does re-ranking this pool at the shipped settings reproduce the frozen prompt?

    A pool is only a valid substrate for parameter analysis if it is a SUPERSET of the evidence
    the published numbers came from. When it is not, the sweep would be measuring a different
    retrieval than the one being compared against, so the answer is recorded in the file rather
    than left for a reader to assume.

    walker (overlap) reproduces exactly — Bhattacharyya is closed-form. walker_interval did not,
    because its sampler was seeded with the built-in hash() of a string, which Python randomises
    per process; the frozen walker_interval evidence came from one unrepeatable draw. The sampler
    is now seeded through hashlib, so pools dumped after that fix are self-consistent, but they
    still cannot match a frozen file produced before it.
    """
    f = PIPE / f"frozen/{ds}/{method}.json"
    # "is this a scored pool?" has to look past records with no candidates — some questions route
    # to no-KG and store an empty list, and testing only records[0] silently skipped the whole
    # check whenever the first question happened to be one of them
    first = next((r["candidates"][0] for r in records if r.get("candidates")), None)
    if not f.exists() or first is None or "cos" not in first:
        return None
    fr = {i["uid"]: (i.get("kg_block") or "").strip() for i in json.load(open(f))["items"]}
    same = diff = 0
    for r in records:
        kg = fr.get(r["uid"])
        if not kg: continue
        if _fmt_top10(r["candidates"]).strip() == kg: same += 1
        else: diff += 1
    return {"matches_frozen": diff == 0, "n_same": same, "n_diff": diff}


def write(ds, method, records, source, params, frozen_check=None):
    out = PIPE / f"pool/{ds}"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{method}.jsonl"
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps({**r, "dataset": ds, "method": method,
                                "source": source, "params": params,
                                "frozen_check": frozen_check},
                               ensure_ascii=False) + "\n")
    n_cand = [len(r["candidates"]) for r in records]
    med = sorted(n_cand)[len(n_cand) // 2] if n_cand else 0
    flag = ""
    if frozen_check is not None:
        flag = ("  ✅ 重現 frozen" if frozen_check["matches_frozen"]
                else f"  ⚠ 與 frozen 不符 {frozen_check['n_diff']}/{frozen_check['n_same']+frozen_check['n_diff']}")
    print(f"  {method:16s} {len(records):4d} 題  候選中位數 {med:4d}  最多 {max(n_cand or [0]):5d}"
          f"  -> pool/{ds}/{method}.jsonl{flag}")


def from_raw_hops(ds, hop):
    f = PIPE / f"frozen/{ds}/raw_{hop}hop.json"
    if not f.exists():
        return None
    items = json.load(open(f))["items"]
    recs = []
    for it in items:
        names = it.get("concepts_retrieved") or []
        recs.append({"uid": it["uid"], "gold": it["gold"], "route": it.get("route"),
                     "n_candidates": len(names),
                     # raw dumps carry no per-candidate score: the ranking IS the traversal order,
                     # which is what makes them the unranked control
                     "candidates": [{"rank": i + 1, "name": nm} for i, nm in enumerate(names)]})
    return recs


def from_chain_cache(ds, method):
    for cand in (PIPE / f"cache/{method}_baseline_{ds}__{MODEL.replace('/','_')}.json",
                 Path(PIPE.parent / f"cache/{method}_baseline_{ds}__{MODEL.replace('/','_')}.json")):
        if cand.exists():
            src = cand
            break
    else:
        return None
    recs_in = json.load(open(src))
    recs_in = recs_in.get("results", recs_in)
    out = []
    for r in recs_in:
        chains = r.get("chains_full") or []
        out.append({"uid": r["uid"], "gold": r.get("gold"), "route": method,
                    "n_candidates": len(chains),
                    "anchors": r.get("anchors") or r.get("seeds"),
                    "candidates": [{"rank": i + 1, "chain": c} for i, c in enumerate(chains)]})
    return out


def from_walker(ds, method):
    # pool_* first: the dump runs under its own WALKER_OUT_TAG so it cannot overwrite the
    # canonical pipeline_* checkpoint that produced the published frozen prompts. pipeline_* is
    # still accepted in case a future run dumps in place.
    for f in (PIPE / f"cache/pool_{ds}_{method}__{MODEL.replace('/','_')}.json",
              PIPE / f"cache/pipeline_{ds}_{method}__{MODEL.replace('/','_')}.json"):
        if f.exists():
            break
    else:
        return None
    recs_in = json.load(open(f))
    recs_in = recs_in.get("results", recs_in)
    if not any("pool" in r for r in recs_in):
        return None            # ran before WALKER_POOL_DUMP existed
    out = []
    for r in recs_in:
        pool = r.get("pool") or []
        out.append({"uid": r["uid"], "gold": r.get("gold"), "route": r.get("route"),
                    "patient_days": r.get("patient_days"),
                    "n_candidates": len(pool), "candidates": pool,
                    "walk_params": r.get("pool_params")})
    return out


for ds in DATASETS:
    print(f"\n### {ds}")
    if WALK:
        dd = PIPE / f"datasets/{ds}"
        for method, bc_mode in [("walker", "overlap"), ("walker_interval", "interval_sample")]:
            print(f"  re-walking {method} (WALKER_POOL_DUMP=1) …", flush=True)
            # eval_kg_walker_full.py directly, NOT build_kg.py: build_kg re-freezes prompts as a
            # side effect and would rewrite the exact files every published number was read from.
            # WALKER_OUT_TAG also keeps the dump out of the canonical checkpoint — whose resume
            # logic would otherwise report "already done → 0 to go" and skip the walk entirely.
            env = dict(os.environ,
                       BENCH_PATH=str(dd / "benchmark.json"), PD_PATH=str(dd / "durations.json"),
                       SEEDS_PATH=str(dd / "seeds.json"), SYM_PATH=str(dd / "symptoms.json"),
                       QE_PATH=str(dd / "query_entities.json"),
                       WALKER_LLM=MODEL, WALKER_MULTI_AXIS="1", WALKER_NO_MDN="0",
                       WALKER_BC_MODE=bc_mode, WALKER_RETRIEVAL_ONLY="1", WALKER_POOL_DUMP="1",
                       WALKER_OUT_TAG=f"pool_{ds}_{method}",
                       WALKER_N_WORKERS=os.environ.get("WALKER_N_WORKERS", "3"))
            subprocess.run([sys.executable, str(PIPE / "code/eval_kg_walker_full.py")],
                           env=env, cwd=str(PIPE))

    for method in ["walker", "walker_interval"]:
        recs = from_walker(ds, method)
        if recs:
            write(ds, method, recs, f"cache/pool_{ds}_{method}", recs[0].get("walk_params"),
                  frozen_check=check_against_frozen(ds, method, recs))
        else:
            print(f"  {method:16s} — 尚無 pool（需以 --walk 重跑檢索）")

    for hop in (1, 2):
        recs = from_raw_hops(ds, hop)
        if recs:
            write(ds, f"raw_{hop}hop", recs, f"frozen/{ds}/raw_{hop}hop.json:concepts_retrieved",
                  {"cap": 50, "prompt_top_k": 10, "scored": False,
                   "replayable": ["top_k"], "needs_rerun": ["cap", "hop", "seeds"]})

    for method in ("tog", "hykge"):
        recs = from_chain_cache(ds, method)
        if recs:
            write(ds, method, recs, f"cache/{method}_baseline_{ds}:chains_full",
                  {"scored": False, "prompt_top_k": 10,
                   "replayable": ["top_k"], "needs_rerun": ["hop", "anchors", "reranker"]})
print()
