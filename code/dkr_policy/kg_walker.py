"""Auto-stopping KG walker.

Step 1 — entity extraction (seeds): provided by caller as list of CUI+name.
Step 2 — BFS walk with score-based auto-stop:
    score(node) = cos_to_query(node) + bc_compat(node, t_obs)
    Expand only if score ≥ min_score.
    max_hops is a safety cap, not a target depth.
    Hubs (out-degree > hub_degree) are not expanded (but may be kept as
    candidates if score passes).
Step 3 — caller packages paths into a KG block for the LLM.

Iteration design rule: keep score simple (additive). Tunable knobs are
min_score, max_hops, hub_degree. BC = 0 for non-disease nodes (drugs,
procedures, findings) — those rank by cos alone.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from umls_neo4j import get_driver
# NOTE: `from retrieve_simple import _node_emb` was removed — the name was never used here.
# That vestigial import was the only thing keeping retrieve_simple.py alive, which in turn kept
# kde_bins.py (KDE/histogram temporal math) and coverage_lookup.py (tiered duration source)
# imported. Both are superseded: the walker scores time via bc_llm_direct (Bhattacharyya /
# interval on LLM-judged log-normal durations), so all three were removed from pipeline/.
# Originals remain in scripts/dkr_policy/. walk() loads embeddings itself via _load_umls_emb().


# Hub filter — out-degree above which a node is NOT expanded.
# Empirically: "Drug product", "Evaluation - action", "Measurement - action"
# have 10K+ outgoing edges. Real clinical concepts ≤ 1000.
DEFAULT_HUB_DEG = 3000


def _node_degree(session, cui: str) -> int:
    """Combined in+out degree (we now walk bidirectionally in _neighbors).
    Used by hub-prune: anatomy hubs like 'Cardiac structure' have huge
    inbound finding_site_of fanout that outgoing-only count would miss."""
    r = session.run(
        "MATCH (a:Concept {CUI: $cui})-[]-(b:Concept) RETURN count(b) AS d",
        cui=cui,
    ).single()
    return r["d"] if r else 0


# Backwards-compat alias.
def _outdegree(session, cui: str) -> int:
    return _node_degree(session, cui)


import re as _re
# Filter SNOMED context-dependent categories ([D], [X], [V], [M], [Q], [SO],
# EXP) and retired concepts. These are hub-like artefacts that cause bad
# expansion: e.g. Migraine NOS → has_definitional_manifestation → [D]Headache
# → definitional_manifestation_of → Tension-type/Thunderclap headache. The
# [D]Headache hub surfaces 50+ unrelated headache variants in 2 hops.
_BAD_PREFIX = _re.compile(r"^\s*\[(d|x|v|q|m|so|exp)\]\s*", _re.IGNORECASE)
_BAD_SUFFIX = _re.compile(r"-?\s*retired\s*-?\s*$", _re.IGNORECASE)

# Veterinary / non-human concepts (UMLS contains many SNOMED-CT-VET entries).
# Observed pollution in walker v7: "Enzootic pneumonia of sheep", "Enzootic
# mycoplasmal pneumonia of swine", "Canine vibriosis", "Foodborne botulism
# type A" (the latter is a strain not a clinical entity).
_VET_RE = _re.compile(
    r"\b("
    r"sheep|swine|porcine|ovine|bovine|cattle|canine|feline|equine|murine|"
    r"avian|poultry|chicken|quail|fish|piscine|reptile|"
    r"enzootic|panzootic|epizootic|"
    r"newcastle\s+disease|foot\s+and\s+mouth"
    r")\b",
    _re.IGNORECASE,
)

# Age-specific concepts that are misleading for adult cases. Conservative:
# only filter the obviously inappropriate ones; "pediatric"/"adult" qualifier
# alone is too aggressive. Walker doesn't know patient age yet; for now these
# are dropped universally because they're almost always background noise in
# adult-dominant bench325. Revisit if peds case fraction grows.
_PEDS_NOISE_RE = _re.compile(
    r"\b("
    r"of\s+infancy|in\s+infancy|infantile|neonatal|"
    r"letterer-?siwe"
    r")\b",
    _re.IGNORECASE,
)


def _is_clinical_concept(name: str) -> bool:
    if not name:
        return False
    if _BAD_PREFIX.search(name):
        return False
    if _BAD_SUFFIX.search(name):
        return False
    if _VET_RE.search(name):
        return False
    if _PEDS_NOISE_RE.search(name):
        return False
    return True


# RELA blocklist — these surfaced false-friend paths in v8/v9 audit.
# - has_finding_method / finding_method_of  : qualifier hub "Listening - action"
#   (aud_012 walker took this to land on Pleural rub instead of Pericardial)
# - has_finding_informer / has_subject_relationship_context / has_temporal_context
#   / has_finding_context: SNOMED context wrappers (not clinical reasoning)
# - has_associated_morphology / associated_morphology_of: "Inflammation" is a
#   massive hub linking every inflammatory disease to every other
# - possibly_equivalent_to / was_a / inverse_was_a / alternative_of:
#   historical/synonym noise that leaks into context-dependent variants
def _bc_weight() -> float:
    return float(os.environ.get("WALKER_BC_WEIGHT", "0.3"))


def _hop_penalty() -> float:
    """μ in score = cos + λ·bc − μ·hop.

    Was a literal 0.08 at the scoring line. Made settable so a parameter sweep can vary it the
    same way it varies λ, and so the pool dump can record which value produced a stored ranking —
    an unrecorded constant is the one knob an analysis silently cannot account for.
    """
    return float(os.environ.get("WALKER_HOP_PENALTY", "0.08"))


_BAD_RELA = frozenset({
    "has_finding_method", "finding_method_of",
    "has_finding_informer", "finding_informer_of",
    "has_subject_relationship_context", "subject_relationship_context_of",
    "has_temporal_context", "temporal_context_of",
    "has_finding_context", "finding_context_of",
    "has_associated_morphology", "associated_morphology_of",
    "possibly_equivalent_to",
    "was_a", "inverse_was_a",
    "alternative_of", "has_alternative",
})


def _neighbors(session, cui: str, limit: int = 500):
    """Return [(neighbor_cui, neighbor_name, rela), ...] of clinical concepts.

    Bidirectional: SNOMED finding→disease edges (e.g. 'pericardial rub'
    has_finding_site Pericardium ← Pericarditis) are only reachable if we
    follow inbound finding_site_of as well. v10 audit showed strict
    outgoing-only missed 'Pericardial friction rub' even though it sits
    2 hops from Pericarditis via Pericardium.

    Filters: clinical-concept name check + RELA blocklist."""
    rows = session.run(
        "MATCH (a:Concept {CUI: $cui})-[r]-(b:Concept) "
        "RETURN b.CUI AS cui, b.name AS name, r.RELA AS rela LIMIT $lim",
        cui=cui, lim=limit,
    ).data()
    out, seen = [], set()
    for r in rows:
        rela = r["rela"]
        if not rela or rela in _BAD_RELA:
            continue
        if not _is_clinical_concept(r["name"]):
            continue
        key = (r["cui"], rela)
        if key in seen:  # bidirectional Cypher may double-emit
            continue
        seen.add(key)
        out.append((r["cui"], r["name"], rela))
    return out


def _suffix_role(name: str) -> str:
    """Suffix-only fallback. Use _role_for(cui, name) instead — it tries
    TUI lookup first. Kept for the rare CUIs missing from cui_to_tui.csv."""
    if not name: return "?"
    nl = name.lower()
    if nl.endswith("(disorder)"):     return "disease"
    if nl.endswith("(finding)"):      return "finding"
    if nl.endswith("(procedure)"):    return "procedure"
    if nl.endswith("(organism)"):     return "organism"
    if nl.endswith("(morphologic abnormality)"): return "morphology"
    if nl.endswith("(substance)"):    return "substance"
    if nl.endswith("(qualifier value)"): return "qualifier"
    if nl.endswith("(situation)"):    return "situation"
    if nl.endswith("(body structure)"): return "anatomy"
    if nl.endswith("(product)"):      return "drug"
    if nl.endswith("(physical object)"): return "object"
    return "other"


def _role_for(cui: str, name: str) -> str:
    """Primary role via UMLS TUI (`tui_roles.get_role`), suffix fallback.
    TUI captures concepts whose canonical names lack SNOMED suffixes
    (e.g. 'Chronic bronchitis', 'Acanthosis nigricans' are T047 diseases
    but have no '(disorder)' tag). Use this everywhere instead of
    _suffix_role."""
    from tui_roles import get_role as _tui_role
    r = _tui_role(cui)
    if r not in ("unknown", "other"):
        return r
    sfx = _suffix_role(name)
    return sfx if sfx not in ("?", "other") else r


def walk(
    seeds: list[tuple[str, str]],
    q_emb,  # np.ndarray OR list of np.ndarray for multi-query (max-cos)
    t_obs: float | None = None,
    bc_fn=None,
    min_score: float = 0.40,
    max_hops: int = 4,
    hub_degree: int = DEFAULT_HUB_DEG,
    neighbor_limit: int = 500,
    verbose: bool = False,
) -> list[dict]:
    _bc_w = _bc_weight()
    _mu = _hop_penalty()
    # Consequence-aware survival (off by default → canonical walk unchanged).
    # A time-compatible node (bc ≥ thr) survives frontier-pruning and expands
    # even if its overall score < min_score, provided cos ≥ a noise floor.
    # This lets the walk reach dangerous, time-compatible diagnoses that are
    # semantically unlike the symptoms (low cos) and would otherwise be pruned.
    _bc_surv = float(os.environ.get("WALKER_BC_SURVIVE", "0") or 0)
    _bc_surv_cos_floor = float(os.environ.get("WALKER_BC_SURVIVE_COS_FLOOR", "0.25") or 0.25)
    """BFS walker with batched encoding.

    Per hop:
      1. From each frontier node, fetch all neighbors from Neo4j (skip hubs).
      2. Collect unique unseen neighbor (CUI, name) pairs.
      3. Batch-encode ALL their embeddings in one SapBERT call → 100× speedup
         vs per-node encoding.
      4. Compute cos to query in one matmul; add BC for disease nodes.
      5. Keep nodes with score ≥ min_score; only those keep expanding next hop.
    """
    from duration_kg_rag import _load_umls_emb
    idx, enc = _load_umls_emb()
    matrix = idx["matrix"]; cuis_idx = idx["cuis"]
    cui2row = {c: i for i, c in enumerate(cuis_idx)}

    visited = {cui for cui, _ in seeds}
    # frontier tuple: (cui, name, hop, path, origin_seed)
    # origin_seed is the seed CUI that started this branch — propagated
    # through BFS so each result can be traced back to "started from this seed".
    frontier = [(cui, name, 0, [], name) for cui, name in seeds]
    results: list[dict] = []
    drv = get_driver()

    # ---- Seeds themselves as hop-0 results -------------------------------
    # Earlier design dropped seeds from results (they were only used as BFS
    # starting points). That caused gold to disappear from kg_block when the
    # gold disease was a seed — e.g. m325_148 had "Chronic bronchitis" as
    # the first DDx seed but LLM never saw it. Now seeds enter results with
    # hop=0 and full score so the LLM sees our highest-confidence entries.
    if seeds:
        seed_emb = np.zeros((len(seeds), matrix.shape[1]), dtype=matrix.dtype)
        need_enc_seed_idx, need_enc_seed_names = [], []
        for i, (cui, name) in enumerate(seeds):
            row = cui2row.get(cui)
            if row is not None:
                seed_emb[i] = matrix[row]
            else:
                need_enc_seed_idx.append(i)
                need_enc_seed_names.append(name or "")
        if need_enc_seed_names:
            ne = enc.encode(need_enc_seed_names)
            ne = ne / np.maximum(np.linalg.norm(ne, axis=1, keepdims=True), 1e-12)
            for k, i in enumerate(need_enc_seed_idx):
                seed_emb[i] = ne[k]
        if isinstance(q_emb, (list, tuple)):
            seed_cos = (seed_emb @ np.stack(q_emb).T).max(axis=1)
        else:
            seed_cos = seed_emb @ q_emb
        for i, (cui, name) in enumerate(seeds):
            bc_s = 0.0
            seed_role = _role_for(cui, name)
            # Fix C: only compute BC for disease-role seeds. Findings (e.g.
            # "Chronic vomiting" from symptom-derived seeds) have no
            # meaningful "typical course" — BC noise on them poisons ranking.
            # Extended BC gate: include finding + organism (drug skipped — sparse coverage).
            # Configurable via WALKER_BC_ROLES, default "disease,finding,organism".
            _bc_roles = set(os.environ.get("WALKER_BC_ROLES", "disease,finding,organism").split(","))
            if bc_fn is not None and t_obs is not None and seed_role in _bc_roles:
                try:
                    bc_s = float(bc_fn(cui, name, t_obs))
                except Exception:
                    bc_s = 0.0
            sc = float(seed_cos[i]) + _bc_w * bc_s  # hop=0, no path-length penalty
            results.append({
                "cui": cui, "name": name,
                "role": seed_role,
                "hop": 0, "score": sc,
                "cos": float(seed_cos[i]), "bc": bc_s,
                "path": [("seed", name, sc)],
                "origin_seed": name,
            })

    with drv.session() as s:
        while frontier:
            # raw[(cui, name)] = list of (src_cui, src_name, src_path, rela, hop, origin_seed)
            raw: dict[tuple[str, str], list] = {}
            for src_cui, src_name, hop, path, origin_seed in frontier:
                if hop >= max_hops:
                    continue
                deg = _node_degree(s, src_cui)
                # Anatomy / process / substance nodes are intermediates with
                # huge fanout (Cardiac structure ~1000+ findings via
                # finding_site_of). Use a tighter degree cap for these so they
                # don't dominate hop-2 expansion. Disease hubs already capped
                # at hub_degree (default 3000) — diseases as intermediates
                # are usually meaningful.
                src_role = _role_for(src_cui, src_name)
                # Anatomy cap 1500 (was 300): Pericardium has deg=2294
                # because it's an essential bridge to pericardial findings;
                # 300 was over-eager and killed the legitimate use case.
                # 1500 still prunes generic 'Cardiac structure' (~5000).
                cap = 1500 if src_role in ("anatomy", "process", "substance") else hub_degree
                if deg > cap:
                    if verbose:
                        print(f"  [hub-prune hop={hop} role={src_role} cap={cap}] {src_name} deg={deg}", flush=True)
                    continue
                for cui, name, rela in _neighbors(s, src_cui, neighbor_limit):
                    if cui in visited:
                        continue
                    raw.setdefault((cui, name), []).append(
                        (src_cui, src_name, path, rela, hop, origin_seed)
                    )
            if not raw:
                break

            # ---- Step B: batch-score all unique neighbors ----
            keys = list(raw.keys())
            # split into pre-indexed (free embedding) vs need-to-encode
            pre_rows = []   # idx in keys → matrix row
            need_enc_idx = []
            need_enc_names = []
            for i, (cui, name) in enumerate(keys):
                row = cui2row.get(cui)
                if row is not None:
                    pre_rows.append((i, row))
                else:
                    need_enc_idx.append(i)
                    need_enc_names.append(name or "")
            # Embeddings array for batch
            emb_arr = np.zeros((len(keys), matrix.shape[1]), dtype=matrix.dtype)
            for i, row in pre_rows:
                emb_arr[i] = matrix[row]
            if need_enc_names:
                ne = enc.encode(need_enc_names)
                ne = ne / np.maximum(np.linalg.norm(ne, axis=1, keepdims=True), 1e-12)
                for k, i in enumerate(need_enc_idx):
                    emb_arr[i] = ne[k]
            # Multi-query support: q_emb can be a single vector or a list.
            # When list, take max cos across queries (permissive: a node passes
            # if any query views it as relevant). Lets symptom-state query and
            # question-intent query both contribute.
            if isinstance(q_emb, (list, tuple)):
                stacked = np.stack(q_emb)  # (Q, D)
                cos_all = emb_arr @ stacked.T  # (N, Q)
                cos_arr = cos_all.max(axis=1)
            else:
                cos_arr = emb_arr @ q_emb  # (N,)

            # BC for disease-role nodes. bc_fn: (cui, name, t) -> float in [0,1].
            bc_arr = np.zeros(len(keys), dtype=float)
            # Precompute roles via TUI lookup once per hop.
            roles = [_role_for(cui, name) for cui, name in keys]
            if bc_fn is not None and t_obs is not None:
                _bc_roles = set(os.environ.get("WALKER_BC_ROLES", "disease,finding,organism").split(","))
                for i, (cui, name) in enumerate(keys):
                    if roles[i] in _bc_roles:
                        try:
                            bc_arr[i] = float(bc_fn(cui, name, t_obs))
                        except Exception:
                            bc_arr[i] = 0.0
            # Score: cos primary, BC secondary multiplier (lowered from 0.5
            # → 0.3 — audit showed bc=0.99 on cos=0.4 noise produced false
            # positives like "Flax-dressers' disease"), minus hop-length
            # penalty (kg_thought.md §5,9: −η·|p| prevents far-from-seed
            # neighbors with coincidental high BC from dominating).
            # The hop_arr indexes [hop_of_src + 1] per neighbor.
            hop_arr = np.array([raw[(cui, name)][0][4] + 1
                                 for cui, name in keys], dtype=float)
            # Score = semantic + temporal − hop penalty. Nothing else: the intent
            # role bonus (+0.15 when a node's role matched the question's intent
            # target) was removed so the utility is exactly cos + λ·bc − μ·hop.
            score_arr = cos_arr + _bc_w * bc_arr - _mu * hop_arr

            # filter + build results + next_frontier
            next_frontier = []
            new_kept = 0
            for i, (cui, name) in enumerate(keys):
                score = float(score_arr[i])
                # Channel-1: relevance survival (score ≥ min_score).
                # Channel-2: consequence/time-compat survival (bc high, cos not noise).
                surv_relevance = score >= min_score
                surv_consequence = (_bc_surv > 0
                                    and float(bc_arr[i]) >= _bc_surv
                                    and float(cos_arr[i]) >= _bc_surv_cos_floor)
                if (not surv_relevance and not surv_consequence) or cui in visited:
                    continue
                visited.add(cui)
                src_cui, src_name, src_path, rela, src_hop, origin_seed = raw[(cui, name)][0]
                new_path = src_path + [(rela, name, score)]
                results.append({
                    "cui": cui, "name": name,
                    "role": roles[i],
                    "hop": src_hop + 1, "score": score,
                    "cos": float(cos_arr[i]), "bc": float(bc_arr[i]),
                    "path": new_path,
                    "origin_seed": origin_seed,
                    "surv": "relevance" if surv_relevance else "consequence",
                })
                new_kept += 1
                if src_hop + 1 < max_hops:
                    next_frontier.append((cui, name, src_hop + 1, new_path, origin_seed))
            if verbose:
                print(f"  hop done: raw={len(keys)} kept_new={new_kept} "
                       f"next_frontier={len(next_frontier)} total={len(results)}",
                       flush=True)
            frontier = next_frontier

    results.sort(key=lambda r: -r["score"])

    # Fix B: reserve top-N slots for highest-scoring seed-diseases.
    # Symptom-finding cos can flood top-K, evicting seed-diseases (the LLM's
    # diagnostic hypotheses for the case). Reserving slots ensures the LLM
    # always sees the candidate disease set, regardless of how high finding
    # nodes score. Set WALKER_SEED_RESERVE=0 to disable.
    _M_RESERVE = int(os.environ.get("WALKER_SEED_RESERVE", "0"))
    if _M_RESERVE > 0:
        seed_disease_idx = [i for i, r in enumerate(results)
                             if r["hop"] == 0 and r["role"] == "disease"]
        # results is already score-sorted, so first M seed-diseases = top-M
        reserved_idx = set(seed_disease_idx[:_M_RESERVE])
        if reserved_idx:
            reserved = [results[i] for i in seed_disease_idx[:_M_RESERVE]]
            others = [r for i, r in enumerate(results) if i not in reserved_idx]
            results = reserved + others
    return results


def format_paths(results: list[dict], top_k: int = 10) -> str:
    """Format walker results as a path block for LLM prompt.
    Path format: 'seed → [rela] → node1 → [rela] → node2 ...'
    """
    if not results:
        return "  (no paths found)"
    lines = []
    for i, r in enumerate(results[:top_k], 1):
        path_str = " → ".join(
            f"[{rela}] {n[:50]}" for rela, n, _ in r["path"]
        )
        role_tag = f" [{r['role']}]" if r['role'] != "?" else ""
        lines.append(f"  {i}. {r['name'][:55]}{role_tag} (score={r['score']:.2f})")
        lines.append(f"     path: {path_str}")
    return "\n".join(lines)


# Encode-query helper so callers don't repeat the same code.
_ENC = None
def _get_enc():
    global _ENC
    if _ENC is None:
        from duration_kg_rag import _load_umls_emb
        _, _ENC = _load_umls_emb()
    return _ENC


def encode_query(phrases: list[str]) -> np.ndarray:
    """Mean-pool unit-norm query embedding from symptom phrases."""
    if not phrases:
        raise ValueError("need at least one phrase")
    enc = _get_enc()
    embs = enc.encode(phrases)
    embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
    q = embs.mean(axis=0)
    q /= max(np.linalg.norm(q), 1e-12)
    return q
