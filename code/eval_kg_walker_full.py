"""Walker retrieval over the UMLS/SNOMED Neo4j KG — the retrieval half of the pipeline.

Per question: build the seed set, encode the symptom query, walk the graph
(kg_walker.walk, score = cos + λ·bc − μ·hop), format the top-K into a kg_block.
Questions with no usable duration/seeds/symptoms fall back to a no-KG route.

Driven by pipeline/build_kg.py, which re-freezes the prompts afterwards.
Set WALKER_RETRIEVAL_ONLY=1 to skip the per-question reader call (build_kg re-runs the
reader separately via run_reader*.py, so the inline call is wasted work during retrieval).
Results are checkpointed every 25 items and the run resumes from the saved file.
"""
import json, sys, os, time, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent          # pipeline/code
ROOT = HERE.parent                               # pipeline
# Self-contained: flat modules in HERE, walker package in HERE/dkr_policy. Nothing outside pipeline/.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "dkr_policy"))
# Walker uses SapBERT-encoded 394K UMLS matrix for cos scoring (medical-domain
# trained, ~6× more CUI coverage than openai-small 61K). Same encoder space
# as seed resolver below ensures cos is comparable across stages.
os.environ.setdefault("DKR_MATRIX_PKL", "cache/umls_broad_embeddings_sapbert.pkl")

CACHE = ROOT / "cache"
# Bench / cache paths overridable via env so we can swap in bench340 without
# code changes. Defaults remain the original bench325 paths.
BENCH_PATH    = os.environ.get("BENCH_PATH",   str(CACHE / "benchmark_pure_tc_325.json"))
PD_PATH       = os.environ.get("PD_PATH",      str(CACHE / "patient_durations_pure_tc_pool_v2.json"))
SEEDS_PATH    = os.environ.get("SEEDS_PATH",   str(CACHE / "benchmark_pure_tc_325_seeds.json"))
SYM_PATH      = os.environ.get("SYM_PATH",     str(CACHE / "benchmark_pure_tc_325_symptoms.json"))
QE_PATH       = os.environ.get("QE_PATH",      str(CACHE / "benchmark_pure_tc_325_query_entities.json"))

bench_list = json.load(open(BENCH_PATH))
pd_c = json.load(open(PD_PATH))
seeds_c = json.load(open(SEEDS_PATH))
sym_c = json.load(open(SYM_PATH))
try:
    qe_c = json.load(open(QE_PATH))
except FileNotFoundError:
    qe_c = {}

# Prompts live ONCE in pipeline/prompts.py (single source of truth). PROMPT_NUMERIC here was
# a verbatim duplicate of prompts.WALKER, PROMPT_NO_KG of prompts.NO_KG; the unused
# WALKER_NARRATIVE / WALKER_HYBRID prompt variants were removed with their formatters.
import importlib.util as _ilu
_pspec = _ilu.spec_from_file_location("P", ROOT / "prompts.py")
_P = _ilu.module_from_spec(_pspec); _pspec.loader.exec_module(_P)
PROMPT = _P.WALKER
PROMPT_NO_KG = _P.NO_KG
NO_DURATION_STR = _P.NO_DURATION_STR

def days_to_phrase(d):
    if isinstance(d, (list, tuple)):
        # Multi-axis: show primary + any distinct secondary axes
        phrases = [days_to_phrase(x) for x in d]
        # Deduplicate by phrase string
        seen = []
        for p in phrases:
            if p not in seen: seen.append(p)
        return " / ".join(seen) if len(seen) > 1 else seen[0]
    if d < 1:    return f"{int(d*24)} hours"
    if d < 14:   return f"{int(d)} days"
    if d < 60:   return f"{int(d/7)} weeks"
    if d < 365:  return f"{int(d/30)} months"
    return f"{int(d/365)} years"


def extract_letter(raw):
    for pattern in [
        r"<a>\s*([A-J])\s*</a>",
        r"\\boxed\{\s*([A-J])\s*\}",
        r"\*\*?Answer:?\*?\*?\s*:?\s*\*?\*?([A-J])\b",
        r"final answer[:\s]+(?:is\s+)?(?:\\boxed\{)?\*?\*?([A-J])\b",
        r"answer is[:\s]+(?:\\boxed\{)?\*?\*?([A-J])\b",
        r"\bthe answer\s*(?:is|:)\s*\*?\*?([A-J])\b",
    ]:
        m = re.search(pattern, raw or "", re.I)
        if m: return m.group(1).upper()
    return None


def format_kg_block(results, top_k=10):
    """Flat top-K, role-tagged. Diseases dominate, with 3 slots reserved for
    non-disease findings so the block keeps some role diversity.

    Intent-aware ordering was removed along with the walker's intent role bonus —
    ranking is now purely the score (cos + λ·bc − μ·hop).

    Format keeps cos/bc breakdown alongside score for post-hoc analysis."""
    if not results:
        return "  (no paths found)"

    disease  = [r for r in results if r["role"] == "disease"]
    non_dis  = [r for r in results if r["role"] != "disease"]
    picked = disease[:7] + non_dis[:3]

    if len(picked) < top_k:
        already = {r["cui"] for r in picked}
        for r in results:
            if r["cui"] in already: continue
            picked.append(r)
            if len(picked) >= top_k: break
    picked.sort(key=lambda r: -r["score"])

    return "\n".join(
        f"  {i}. [{r['role']}] {r['name'][:60]} "
        f"(score={r['score']:.2f}: cos={r['cos']:.2f}+bc={r['bc']:.2f})"
        for i, r in enumerate(picked[:top_k], 1)
    )


def _resolve_concept(name, max_k=2, min_cos=0.70, min_jaccard=0.30):
    """Exact CSV match first (conf=1.0), SapBERT cos fallback otherwise.

    Token-overlap floor on SapBERT hits prevents pathological matches
    like 'Staphylococcus epidermidis' → 'Macrococcus epidermidis'
    (cos=0.91 but wrong genus). For short queries (≤3 tokens) with
    cos < 0.85, require ALL query tokens in the matched term."""
    from lexical_lookup import (
        lookup_exact, lookup_sapbert, _tokens_for_overlap,
    )
    from kg_walker import _is_clinical_concept as _good
    exact = lookup_exact(name)
    out, seen = [], set()
    for cui, _g, term in exact:
        if cui in seen or not _good(term): continue
        out.append((cui, term, 1.0)); seen.add(cui)
        if len(out) >= max_k: break
    if len(out) >= max_k:
        return out
    q_toks = _tokens_for_overlap(name)
    for cui, _g, term, cos in lookup_sapbert(
            name, top_k=max_k * 5, min_cos=min_cos):
        if cui in seen or not _good(term): continue
        if len(q_toks) >= 2:
            t_toks = _tokens_for_overlap(term)
            inter = q_toks & t_toks
            if not inter: continue
            if len(q_toks) <= 3 and cos < 0.85:
                if inter != q_toks: continue
            else:
                jacc = len(inter) / max(len(q_toks | t_toks), 1)
                if jacc < min_jaccard: continue
        out.append((cui, term, float(cos))); seen.add(cui)
        if len(out) >= max_k: break
    return out


def _collect_all_seeds(uid, item, seeds_c, sym_c, max_total=14):
    """DKR Stage 1 seed union: DDx + query_entities (comorbidities, drugs,
    procedures, labs) + symptoms. NO option text — would leak answer axis.

    The query_entities cache is load-bearing for the mechanism / drug /
    procedure axis: aud_068 needs `drugs: methotrexate` to reach
    drug-induced pulmonary disease; aud_197 needs `procedures: mitral
    valve replacement` to reach prosthetic-valve endocarditis."""
    qe = qe_c.get(uid, {}) or {}
    sources = [
        ('ddx',       (seeds_c.get(uid, {}) or {}).get('seeds', [])[:5]),
        ('q_disease', (qe.get('diseases_mentioned') or [])[:4]),
        ('q_drug',    (qe.get('drugs') or [])[:3]),
        ('q_proc',    (qe.get('procedures') or [])[:3]),
        ('q_lab',     (qe.get('lab_findings') or [])[:4]),
        ('finding',   (sym_c.get(uid, {}) or {}).get('symptoms', [])[:5]),
    ]
    seeds, seen = [], set()
    for tag, names in sources:
        for name in names:
            for cui, term, conf in _resolve_concept(name, max_k=1):
                if cui in seen: continue
                seeds.append({'cui': cui, 'name': term, 'src': tag, 'conf': conf})
                seen.add(cui)
                if len(seeds) >= max_total: return seeds
                break  # one anchor per name in rank-1 pass
    # rank-2 pass on DDx for breadth when caps not hit
    for name in (seeds_c.get(uid, {}) or {}).get('seeds', [])[:5]:
        for cui, term, conf in _resolve_concept(name, max_k=2):
            if cui in seen: continue
            seeds.append({'cui': cui, 'name': term, 'src': 'ddx2', 'conf': conf})
            seen.add(cui)
            if len(seeds) >= max_total: return seeds
    return seeds


def _make_query_embedding(item, sym_phrases):
    """Single q_sym mean-pool. Q1+Q1-rev showed q_int (last-sentence intent)
    surfaces meta-concepts and q_all (mean-pool all entities) lets PMH/meds
    hijack the centroid. Entities reach the walker as SEEDS via
    _collect_all_seeds, which is the right channel for them."""
    from kg_walker import encode_query
    if not sym_phrases:
        return None
    return encode_query(sym_phrases)


def _run_no_kg(item, opts, route):
    """CoT-only fallback when duration or seeds are missing."""
    from llm_client import call_llm
    opts_block = "\n".join(f"  {k}. {v}" for k, v in sorted(opts.items()))
    prompt = PROMPT_NO_KG.format(question=item["question"], options_block=opts_block)
    # Same retrieval-only skip as the walker path: build_kg discards the prediction and
    # re-runs the reader separately, so this call is wasted work during retrieval.
    if os.environ.get("WALKER_RETRIEVAL_ONLY", "0") == "1":
        raw, pred = "", None
    else:
        raw = call_llm(prompt, model=os.environ.get("WALKER_LLM", "gpt-5.4-mini"))
        pred = extract_letter(raw)
    return {"uid": item["uid"], "gold": item["answer"], "predicted": pred,
            "is_correct": pred == item["answer"], "route": route,
            "kg_block": "", "raw_response": raw}


_BC_DUR_CACHE = None


def _bc_hybrid(cui, name, t):
    """Hybrid BC: sharp cache lookup where available, MDN K=2 fallback for
    the long tail of UMLS disorders not in the LLM-judged cache.

    Fix A (multi-axis): t can be a single float OR a list of floats.
    When list, compute BC against EACH axis (e.g., current_episode +
    disease_course + past_medical_history) and return MAX. This avoids
    the m325_971-style failure where a single patient_days (current 4-week
    episode) misses the diagnostic anchor on a different axis (3-year
    recurrent course → persistent depressive disorder).

    WALKER_ROLE_AXES=1 (new): replace single t with entity-role-conditional
    t selection. For each entity CUI, look up TUI role and pick axis from
    patient case (organism→exposure_to_onset, drug→post_medication, etc.).

    Q3 audit (mdn_k2_training_defects_audit.md) showed MDN K=2 is compressed
    (mean BC ~0.5) — limits discrimination but strictly beats 0 for cache-miss
    diseases. MDN training defects are a separate fix item."""
    global _BC_DUR_CACHE
    from bc_llm_direct import bc_for_cui, _load_durations
    if _BC_DUR_CACHE is None:
        _BC_DUR_CACHE = _load_durations()

    # Normalize to list of times
    if isinstance(t, (int, float)):
        times = [float(t)]
    elif isinstance(t, (list, tuple)):
        times = [float(x) for x in t if x and x > 0]
    else:
        return 0.0
    if not times:
        return 0.0

    # WALKER_NO_MDN=1: return 0 on cache miss (pure LLM-judged BC, no MDN fallback).
    # Assumes bulk_generate_ondemand was run so cache covers all expected CUIs.
    no_mdn = os.environ.get("WALKER_NO_MDN", "0") == "1"

    # WALKER_PURE_MDN=1: use the MDN K=2 model for EVERY disease (ignore the LLM-judged
    # cache entirely) — pure-MDN ablation, no LLM durations at all.
    pure_mdn = os.environ.get("WALKER_PURE_MDN", "0") == "1"

    def _bc_at(t_val):
        # cache hit → LLM-judged duration cache
        if cui in _BC_DUR_CACHE:
            return bc_for_cui(cui, t_val)
        if not name:
            return 0.0
        # WALKER_BC_CACHE_ONLY=1: skip on-demand LLM generation, bc=0 on cache miss. bc still
        # active on the ~89% of candidates already cached; used to keep a single-question run
        # (the one-shot example) fast instead of firing dozens of sequential LLM calls.
        if os.environ.get("WALKER_BC_CACHE_ONLY", "0") == "1":
            return 0.0
        # cache MISS (new/unseen entity) → let the LLM re-judge its duration on demand
        # (was the MDN K=2 fallback; MDN dropped). bc_ondemand generates {role,min,max}
        # via the LLM, appends to per_disease_durations_on_demand.jsonl, and returns BC.
        try:
            from bc_ondemand import bc_ondemand
            from kg_walker import _role_for
            bc, _src = bc_ondemand(cui, name, t_val, _role_for(cui, name))
            return bc
        except Exception:
            return 0.0

    return max(_bc_at(t_val) for t_val in times)


def process_one(item):
    from kg_walker import walk, _bc_weight, _hop_penalty
    from llm_client import call_llm

    uid = item["uid"]; gold = item["answer"]; opts = item["options"]
    # WALKER_FORCE_NO_KG=1: no-retrieval baseline through the IDENTICAL harness
    # (same model / PROMPT_NO_KG / answer extraction) — only KG evidence differs.
    if os.environ.get("WALKER_FORCE_NO_KG", "0") == "1":
        return _run_no_kg(item, opts, "vanilla_nokg")

    pd_entry = pd_c.get(uid, {}) or {}
    t_primary = pd_entry.get("days")
    has_dur = isinstance(t_primary, (int, float)) and t_primary > 0

    if has_dur:
        # Fix A (multi-axis): collect primary CHIEF_COMPLAINT/ONSET (the main
        # axis) PLUS past_medical_history axes that are materially LONGER (>2×
        # primary). Past_medical_history captures "3-year recurrent course"
        # type information (m325_971 persistent depressive disorder) without
        # introducing acute-tail noise (aud_070 5-day facial onset is on the
        # SAME current chief-complaint axis, not a disease-course axis).
        # BC scorer takes MAX across axes per candidate.
        if os.environ.get("WALKER_MULTI_AXIS", "1") == "1":
            axes_days = [t_primary]
            for span in (pd_entry.get("spans") or []):
                if span.get("role") != "past_medical_history":
                    continue
                d = span.get("days")
                if d and d > t_primary * 2 and d <= 365 * 100:
                    axes_days.append(d)
            axes_days = sorted(set(round(d, 3) for d in axes_days))
            t = axes_days if len(axes_days) > 1 else t_primary
        else:
            t = t_primary
    else:
        # No usable symptom duration -> DEGRADE, don't bail out. Previously these fell back
        # to a plain CoT prompt with no KG at all (617/1273 on the full set). Now they still
        # retrieve, with t_obs=None so walk() skips bc entirely and the utility degrades to
        #     score = cos - 0.08*hop      (semantic + hop penalty, no temporal term)
        t = None

    seed_dicts = _collect_all_seeds(uid, item, seeds_c, sym_c, max_total=14)
    if not seed_dicts:
        return _run_no_kg(item, opts, "no_seeds")

    seeds = [(s['cui'], s['name']) for s in seed_dicts]
    # Query = mean-pool of ALL presenting symptoms (the [:5] salience cap was removed —
    # it was an untuned convention that truncated ~half the set; see design note).
    q_sym = _make_query_embedding(item, sym_c.get(uid, {}).get("symptoms", []))
    if q_sym is None:
        return _run_no_kg(item, opts, "no_symptoms")

    results = walk(seeds, q_sym, t_obs=t, bc_fn=_bc_hybrid,
                   min_score=0.40, max_hops=2, neighbor_limit=200, verbose=False)

    kg_block = format_kg_block(results, top_k=10)
    opts_block = "\n".join(f"  {k}. {v}" for k, v in sorted(opts.items()))
    # Be explicit when there is no duration, so the reader doesn't misread the all-zero bc
    # column as "every candidate is a non-disease".
    dur_str = days_to_phrase(t) if has_dur else NO_DURATION_STR
    prompt = PROMPT.format(question=item["question"], options_block=opts_block,
                           patient_dur_str=dur_str, kg_block=kg_block)
    # Canary: silent kg_block drop bug (llm_seeds NameError) once swallowed
    # kg_block entirely. Don't remove.
    assert kg_block in prompt, f"BUG: kg_block missing from prompt for {uid}"

    # WALKER_RETRIEVAL_ONLY=1: skip the per-question reader call. build_kg only needs
    # kg_block/route/patient_days and re-runs the reader separately (run_reader/run_qwen),
    # so the inline reader here is wasted work — skipping it is what makes re-retrieval fast.
    if os.environ.get("WALKER_RETRIEVAL_ONLY", "0") == "1":
        pred, raw = None, ""
    else:
        raw = call_llm(prompt, model=os.environ.get("WALKER_LLM", "gpt-5.4-mini"))
        pred = extract_letter(raw)
    rec = {"uid": uid, "gold": gold, "predicted": pred,
           "is_correct": pred == gold,
           "route": "walker_kg" if has_dur else "walker_kg_nodur",
           "patient_days": t, "n_walker_candidates": len(results),
           "top1_cos": results[0]["cos"] if results else 0.0,
           "kg_block": kg_block, "raw_response": raw}

    # WALKER_POOL_DUMP=1 keeps the whole retrieved pool with its RAW components, not just the
    # top-K that reached the prompt. Storing cos/bc/hop separately is what makes a later sweep
    # over top-K, λ, μ or the utility formula a re-ranking of this file rather than a re-walk of
    # Neo4j — the same reason kg_block is frozen apart from prompt, pushed one level down.
    #
    # It does NOT make τ (min_score) or max_hops replayable: both gate EXPANSION during the walk,
    # so a lower τ would have reached nodes that were never scored and cannot appear here. Those
    # two still need a fresh run. The values in effect are recorded so a later analysis can tell
    # which knobs its pool actually supports.
    if os.environ.get("WALKER_POOL_DUMP", "0") == "1":
        cap = int(os.environ.get("WALKER_POOL_CAP", "0"))       # 0 = keep everything
        pool = results[:cap] if cap else results
        rec["pool"] = [{"rank": i + 1, "cui": r["cui"], "name": r["name"], "role": r["role"],
                        "hop": r["hop"], "cos": round(r["cos"], 6), "bc": round(r["bc"], 6),
                        "score": round(r["score"], 6),
                        "origin_seed": r.get("origin_seed"),
                        "path": r.get("path"), "surv": r.get("surv")}
                       for i, r in enumerate(pool)]
        rec["pool_params"] = {"lambda_bc": _bc_weight(), "mu_hop": _hop_penalty(),
                              "tau_min_score": 0.40, "max_hops": 2, "neighbor_limit": 200,
                              "bc_mode": os.environ.get("WALKER_BC_MODE", "overlap"),
                              "prompt_top_k": 10, "pool_cap": cap or None,
                              "replayable": ["top_k", "lambda_bc", "mu_hop", "utility_fn",
                                             "role_quota"],
                              "needs_rerun": ["tau_min_score", "max_hops", "neighbor_limit",
                                              "seeds", "query_embedding"]}
    return rec


def _ondemand_lines():
    """Line count of the on-demand duration cache = a running proxy for LLM generation cost.
    Every appended line is one gpt call. A dual-cache path bug once made this grow by ~one call
    PER CANDIDATE PER RUN (402k wasted regenerations); surfacing the rate makes that impossible
    to miss again."""
    try:
        from bc_ondemand import ONDEMAND
        return sum(1 for _ in open(ONDEMAND)) if ONDEMAND.exists() else 0
    except Exception:
        return 0


def main():
    out = {"results": [], "metadata": {"n_total": len(bench_list)}}
    t_start = time.time()
    _gen0 = _ondemand_lines()   # LLM-generation baseline; growth during this run = new gpt calls

    # Use process pool for parallelism (each walker call independent)
    N_WORKERS = int(os.environ.get("WALKER_N_WORKERS", "12"))
    model_tag = os.environ.get("WALKER_LLM", "gpt-5.4-mini").replace(":", "_").replace("/", "_")
    save_tag = os.environ.get("WALKER_OUT_TAG", "bench325_walker_full")
    save_path = CACHE / f"{save_tag}__{model_tag}.json"

    # RESUME: a long retrieval (on-demand duration generation can run for hours) must not
    # lose everything on a crash/reboot. Reload whatever was already saved and skip those
    # uids; combined with periodic checkpointing below this makes the run restartable.
    done = {}
    if save_path.exists():
        try:
            done = {r["uid"]: r for r in json.load(open(save_path)).get("results", [])}
        except Exception:
            done = {}
    if done:
        out["results"] = list(done.values())
        print(f"[resume] {len(done)} already done → {len(bench_list)-len(done)} to go", flush=True)
    todo = [it for it in bench_list if it["uid"] not in done]

    def _checkpoint():
        json.dump(out, open(save_path, "w"), indent=2)

    completed = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        future_to_uid = {ex.submit(process_one, item): item["uid"] for item in todo}
        for fut in as_completed(future_to_uid):
            uid = future_to_uid[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  ERR {uid}: {e}", flush=True)
                continue
            out["results"].append(res)
            completed += 1
            mark = "✓" if res.get("is_correct") else "✗"
            r_route = res.get("route", "?")
            if completed % 25 == 0:
                _checkpoint()
            if completed % 10 == 0 or completed <= 5:
                n_ok = sum(1 for r in out["results"] if r.get("is_correct"))
                elapsed = time.time() - t_start
                eta = elapsed / completed * (len(todo) - completed) / 60
                gens = _ondemand_lines() - _gen0            # gpt duration-gen calls so far
                gpq = gens / completed                       # per question — should be small (<~15)
                warn = "  ⚠ HIGH gen/q — cache not being reused?" if gpq > 40 else ""
                print(f"[{len(out['results'])}/{len(bench_list)}] {uid} ({r_route}) → {res.get('predicted')} {mark} | acc {n_ok}/{len(out['results'])} = {100*n_ok/len(out['results']):.1f}% | LLM-gen {gens} ({gpq:.0f}/q){warn}  ETA {eta:.1f}min", flush=True)

    out["metadata"]["elapsed_sec"] = time.time() - t_start
    out["metadata"]["n_correct"] = sum(1 for r in out["results"] if r.get("is_correct"))
    out["metadata"]["n_run"] = len(out["results"])
    out["metadata"]["accuracy"] = 100 * out["metadata"]["n_correct"] / max(1, out["metadata"]["n_run"])
    from collections import Counter
    out["metadata"]["routes"] = dict(Counter(r.get("route") for r in out["results"]))
    json.dump(out, open(save_path, "w"), indent=2)
    print(f"\nSaved → {save_path}", flush=True)
    print(f"n_correct={out['metadata']['n_correct']}/{out['metadata']['n_run']} "
           f"acc={out['metadata']['accuracy']:.2f}%  routes={out['metadata']['routes']}  "
           f"elapsed={out['metadata']['elapsed_sec']/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
