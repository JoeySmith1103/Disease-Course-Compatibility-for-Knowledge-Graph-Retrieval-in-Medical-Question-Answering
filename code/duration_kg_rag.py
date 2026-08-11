"""Duration-Aware KG-RAG retrieval — M1 to M4 minimal pipeline.

This is the framework instantiated as a single retrieve() function:
  Input:  symptom phrases + t_obs + patient_role
  Output: ranked candidate paths with per-path state labels

Story being tested:
  Duration controls retrieval at three levels:
    (A) WHICH disease nodes to retrieve  (M2 channel-T)
    (B) HOW DEEP / WHICH PATHS to expand (M3 per-path policy)
    (C) WHAT to package for the LLM       (M4 evidence package)

If retrieval set changes meaningfully when t changes (with the same
symptom), the framework's core claim is validated regardless of
downstream MCQ accuracy.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Encoders only. The MDN (SpectrumMDN / PATIENT_TO_NCBI_ROLE_PRIOR) was dropped when bc moved to
# all-LLM durations; retrieve()/_load_mdn()/temporal_compat() went with it.
from encoders import make_encoder  # noqa: E402

CACHE = HERE.parent / "cache"


# ── State classification (Module 3) ─────────────────────────────────────
def classify_state(q: float) -> str:
    if q is None:
        return "NO_SIGNAL"
    if 0.05 <= q <= 0.95:
        return "COMPATIBLE"
    if (0.01 <= q < 0.05) or (0.95 < q <= 0.99):
        return "BORDERLINE"
    return "CONFLICTING"


def compat_score(q: Optional[float]) -> float:
    if q is None:
        return 0.0
    return 1.0 - 2.0 * abs(q - 0.5)


# ── Lazy loaders ─────────────────────────────────────────────────────────
_DA_INDEX = None
_FD_INDEX = None
_TR_INDEX = None         # temporal-relation index (Extension B)
_UMLS_EMB = None
_ENCODER = None


def _load_da_index():
    global _DA_INDEX
    if _DA_INDEX is None:
        with open(CACHE / "disease_anchor_index.json") as f:
            _DA_INDEX = json.load(f)
    return _DA_INDEX


_FD_INDEX_VARIANT = "default"  # "default" or "plus"


def _load_fd_index():
    """Finding→disease index (symptom-mediated retrieval entry).

    When `_FD_INDEX_VARIANT == "plus"`, loads the F+ index that adds
    due_to / associated_with / interprets RELAs (~5K extra edges).
    """
    global _FD_INDEX
    if _FD_INDEX is None:
        fname = ("finding_disease_index_plus.json"
                 if _FD_INDEX_VARIANT == "plus"
                 else "finding_disease_index.json")
        with open(CACHE / fname) as f:
            _FD_INDEX = json.load(f)
    return _FD_INDEX


def use_fd_index_plus():
    """Switch the F-channel index to the plus variant. Resets the cache."""
    global _FD_INDEX, _FD_INDEX_VARIANT
    _FD_INDEX = None
    _FD_INDEX_VARIANT = "plus"


def _load_umls_emb():
    global _UMLS_EMB, _ENCODER
    if _UMLS_EMB is None:
        # Allow env override so we can A/B between matrices (61K openai-small
        # vs 394K SapBERT). Default is the legacy pickle for backwards-compat.
        path = os.environ.get(
            "DKR_MATRIX_PKL",
            str(CACHE / "umls_broad_embeddings.pkl"),
        )
        with open(path, "rb") as f:
            _UMLS_EMB = pickle.load(f)
        _ENCODER = make_encoder(_UMLS_EMB["encoder"])
    return _UMLS_EMB, _ENCODER


def _load_tr_index():
    """Temporal-relation index. Built from SNOMED RELAs:
      occurs_before/after, temporally_followed_by/follows, has_onset, during.
    Maps disease_cui → {is_acute_marker, is_chronic_marker, onset[]}.
    """
    global _TR_INDEX
    if _TR_INDEX is None:
        path = CACHE / "temporal_relation_index.json"
        if path.exists():
            with open(path) as f:
                _TR_INDEX = json.load(f)["disease_temporal"]
        else:
            _TR_INDEX = {}
    return _TR_INDEX


# ── M2 channel-T: temporal compatibility query ──────────────────────────


# ── M2 channel-S: semantic retrieval via UMLS broad embeddings ──────────
def semantic_retrieve(symptom_phrases: list[str], top_k: int = 30
                      ) -> list[dict]:
    """Direct (disorder) match: when symptom phrase contains a disease name.

    For "patient has Crohn disease for 6 weeks", direct cosine match to
    "Crohn disease (disorder)" works. For pure-symptom queries it returns
    weak matches; finding-mediated retrieval handles those.
    """
    idx, enc = _load_umls_emb()
    if not symptom_phrases:
        return []
    embs = enc.encode(symptom_phrases)
    embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
    query = embs.mean(axis=0)
    cosines = idx["matrix"] @ query
    order = np.argsort(-cosines)
    out = []
    for i in order:
        if "(disorder)" not in idx["names"][i]:
            continue
        out.append({
            "cui": idx["cuis"][i], "name": idx["names"][i],
            "cosine": float(cosines[i]),
        })
        if len(out) >= top_k:
            break
    return out


def finding_mediated_retrieve(symptom_phrases: list[str],
                              top_k_findings: int = 15,
                              top_k_per_finding: int = 8,
                              max_finding_breadth: int = 100,
                              top_k_diseases: int = 60) -> list[dict]:
    """Symptom-mediated retrieval: lexical match against finding-role
    concepts in cui_with_terms.csv (entry point), then traverse to
    diseases via {definitional_manifestation_of, associated_finding_of,
    interpretation_of, cause_of, clinical_course_of}.

    Lexical-first design (vs SapBERT-cosine on full 394K matrix): a
    symptom phrase like "abdominal pain" should anchor to its precise
    SNOMED CUI (C0000737), not the cosine-nearest of all concepts.
    SapBERT cosine over the FINDING-subset of the matrix is used as
    fallback only when lexical lookup fails (e.g. "shortness of breath"
    has no canonical row but maps to "Breathlessness" semantically).

    This is the proper SNOMED entry path for symptom-described patients.
    """
    fd = _load_fd_index()
    if not symptom_phrases:
        return []

    # Lexical entry: map each phrase to finding CUIs
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lexical_lookup import lookup_by_role  # noqa: E402

    findings: list[dict] = []
    seen_f = set()
    unmatched: list[str] = []
    for phrase in symptom_phrases:
        hits = lookup_by_role(phrase, allowed_roles=("finding",), max_results=5)
        if not hits:
            unmatched.append(phrase)
            continue
        for cui, _grp, term, _role in hits:
            if cui in seen_f or cui not in fd["finding_to_diseases"]:
                continue
            seen_f.add(cui)
            findings.append({"f_cui": cui, "f_name": term, "cosine": 1.0,
                             "src": "lexical"})

    # SapBERT fallback for unmatched phrases — but ONLY against finding-role
    # rows of the matrix (much narrower than full 394K).
    if unmatched and len(findings) < top_k_findings:
        idx, enc = _load_umls_emb()
        # Build/cache finding-row mask once
        global _FINDING_ROW_MASK, _FINDING_MATRIX, _FINDING_CUIS, _FINDING_NAMES
        if "_FINDING_ROW_MASK" not in globals() or _FINDING_ROW_MASK is None:
            roles = idx.get("role")
            if roles is None:
                # Legacy 61K openai matrix — no role; fall back to fd index keys
                in_fd = set(fd["finding_to_diseases"].keys())
                mask = np.array([c in in_fd for c in idx["cuis"]])
            else:
                mask = np.array([r == "finding" for r in roles])
            _FINDING_ROW_MASK = mask
            _FINDING_MATRIX = idx["matrix"][mask]
            _FINDING_CUIS = [c for c, m in zip(idx["cuis"], mask) if m]
            _FINDING_NAMES = [n for n, m in zip(idx["names"], mask) if m]
        embs = enc.encode(unmatched)
        embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
        query = embs.mean(axis=0)
        cos_f = _FINDING_MATRIX @ query
        for i in np.argsort(-cos_f):
            cui = _FINDING_CUIS[i]
            if cui in seen_f or cui not in fd["finding_to_diseases"]:
                continue
            seen_f.add(cui)
            findings.append({
                "f_cui": cui, "f_name": _FINDING_NAMES[i],
                "cosine": float(cos_f[i]), "src": "sapbert",
            })
            if len(findings) >= top_k_findings:
                break

    findings = findings[:top_k_findings]

    # Traverse to diseases
    disease_score: dict[str, float] = {}
    disease_via: dict[str, list] = {}
    disease_name: dict[str, str] = {}
    for f in findings:
        connected = fd["finding_to_diseases"].get(f["f_cui"], [])
        # Skip super-broad findings (connect to >max_finding_breadth diseases)
        if len(connected) > max_finding_breadth:
            connected = connected[:max_finding_breadth]
        # Specificity = 1 / |connected| — prefer specific findings
        specificity = 1.0 / max(len(connected), 1)
        for c in connected[:top_k_per_finding]:
            d_cui = c["d_cui"]
            disease_name[d_cui] = c["d_name"]
            # Score: cosine_to_finding × specificity_of_finding
            contribution = f["cosine"] * specificity
            disease_score[d_cui] = disease_score.get(d_cui, 0.0) + contribution
            disease_via.setdefault(d_cui, []).append({
                "f_cui": f["f_cui"], "f_name": f["f_name"],
                "rela": c["rela"], "cosine": f["cosine"],
            })

    ranked = sorted(disease_score.items(), key=lambda x: -x[1])[:top_k_diseases]
    out = []
    for d_cui, score in ranked:
        out.append({
            "cui": d_cui, "name": disease_name[d_cui],
            "score": score,
            "via_findings": disease_via[d_cui][:3],  # explainability
        })
    return out


# ── M2 channel-D: D—A—D bipartite expansion ─────────────────────────────
def bipartite_expand(seed_cuis: list[str], max_per_anchor: int = 50,
                     max_anchor_breadth: int = 200,
                     top_k: int = 50,
                     anchor_weight_fn=None) -> list[dict]:
    """Expand seed diseases through shared anchors to find co-anchored diseases.

    For each seed disease:
      seed → anchors (limit to top max_anchor_breadth most-specific anchors)
            → diseases sharing those anchors

    edge_weight(anchor) = specificity × duration_coherence(anchor)
        specificity = 1 / |diseases sharing anchor|
        duration_coherence = anchor_weight_fn(anchor_cui, anchor_diseases)
                             ∈ (0, ∞]; defaults to 1 when fn is None
                             (= legacy duration-blind walk).

    The duration-coherence factor is what makes the graph WALK itself
    duration-conditioned: anchors whose attached diseases are mostly
    temporally implausible at e_t lose edge weight, so the walk
    preferentially follows temporally-coherent neighborhoods.
    """
    idx = _load_da_index()
    d2a = idx["disease_to_anchors"]
    a2d = idx["anchor_to_diseases"]

    co_score: dict[str, float] = {}
    co_path_count: dict[str, int] = {}
    co_via: dict[str, list] = {}

    for seed in seed_cuis:
        anchors = d2a.get(seed, [])
        # Filter to non-broad anchors first (skip "Lung", "Pain" etc.)
        spec_anchors = sorted(
            anchors,
            key=lambda a: len(a2d.get(a["anchor_cui"], {}).get("diseases", [])),
        )[:max_anchor_breadth]
        for a in spec_anchors:
            anchor_pool = a2d.get(a["anchor_cui"], {}).get("diseases", [])
            if len(anchor_pool) > max_per_anchor:
                anchor_pool = anchor_pool[:max_per_anchor]
            specificity = 1.0 / max(len(a2d.get(a["anchor_cui"], {}).get("diseases", [])), 1)
            duration_w = 1.0
            if anchor_weight_fn is not None:
                duration_w = anchor_weight_fn(a["anchor_cui"], tuple(anchor_pool))
            edge_w = specificity * duration_w
            for d_cui in anchor_pool:
                if d_cui == seed:
                    continue
                co_score[d_cui] = co_score.get(d_cui, 0.0) + edge_w
                co_path_count[d_cui] = co_path_count.get(d_cui, 0) + 1
                co_via.setdefault(d_cui, []).append(
                    {"seed": seed, "anchor_cui": a["anchor_cui"],
                     "anchor_name": a["anchor_name"], "anchor_type": a["anchor_type"],
                     "duration_coherence": duration_w})

    ranked = sorted(co_score.items(), key=lambda x: -x[1])[:top_k]
    out = []
    for d_cui, s in ranked:
        out.append({
            "cui": d_cui, "score": s, "n_paths": co_path_count[d_cui],
            "via": co_via[d_cui][:5],  # keep top-5 paths for explainability
        })
    return out


# ── M3: per-candidate state + policy action dispatch ────────────────────
@dataclass
class RetrievalCandidate:
    cui: str
    name: str
    sources: set = field(default_factory=set)        # {"L", "S", "F", "T", "D"}
    cosine: float = 0.0
    q: Optional[float] = None
    compat: float = 0.0
    log_p: Optional[float] = None    # log g(log t_obs | d) — framework's per-disease duration likelihood
    state: str = "NO_SIGNAL"
    tr_marker: Optional[str] = None  # SNOMED temporal RELA marker: "acute" | "chronic" | None
    bipartite_score: float = 0.0
    bipartite_paths: int = 0
    via_anchors: list = field(default_factory=list)
    finding_score: float = 0.0
    via_findings: list = field(default_factory=list)
    via_llm_seed: Optional[str] = None     # canonical seed name when source==L
    rank_sem: int = 999
    rank_finding: int = 999
    rank_temp: int = 999
    rank_anchor: int = 999
    rrf_score: float = 0.0
    action: str = ""

    def to_dict(self):
        return {
            "cui": self.cui, "name": self.name,
            "sources": sorted(self.sources),
            "cosine": round(self.cosine, 3),
            "q": round(self.q, 3) if self.q is not None else None,
            "compat": round(self.compat, 3),
            "log_p": round(self.log_p, 3) if self.log_p is not None else None,
            "state": self.state, "action": self.action,
            "tr_marker": self.tr_marker,
            "bipartite_score": round(self.bipartite_score, 4),
            "bipartite_paths": self.bipartite_paths,
            "finding_score": round(self.finding_score, 4),
            "via_anchors": [
                {"name": v.get("anchor_name"), "type": v.get("anchor_type")}
                for v in self.via_anchors[:3]
            ],
            "via_findings": [
                {"name": v.get("f_name"), "rela": v.get("rela")}
                for v in self.via_findings[:3]
            ],
            "via_llm_seed": self.via_llm_seed,
            "rrf_score": round(self.rrf_score, 4),
            "ranks": {"sem": self.rank_sem, "finding": self.rank_finding,
                      "temp": self.rank_temp, "anchor": self.rank_anchor},
        }


def assign_action(state: str, has_anchor_evidence: bool) -> str:
    """Module 3 policy dispatcher.

    A1 = expand_compatible (default for COMPATIBLE)
    A2 = preserve_borderline (BORDERLINE — keep but flag atypical)
    A3 = downweight_conflicting (CONFLICTING — drop unless rescued by anchor)
    A4 = ignore_no_signal (NO_SIGNAL or AXIS_MISMATCH — don't apply policy)
    """
    if state == "COMPATIBLE":
        return "A1_expand_compatible"
    if state == "BORDERLINE":
        return "A2_preserve_borderline"
    if state == "CONFLICTING":
        if has_anchor_evidence:
            return "A2_preserve_borderline"  # rescue via non-temporal evidence
        return "A3_downweight_conflicting"
    return "A4_no_signal"


# ── Pipeline orchestration ──────────────────────────────────────────────
def llm_seed_retrieve(seed_disease_names: list[str], top_k_per_seed: int = 3
                      ) -> list[dict]:
    """Resolve LLM-generated disease seeds to SNOMED CUIs.

    Lexical-first against cui_with_terms.csv (filtered to candidate roles
    disease/organism/finding/drug/procedure), with SapBERT cosine on the
    candidate-role subset of the matrix as fallback for unmatched seeds.

    Replaces the legacy ``"(disorder)" in name`` substring filter, which
    silently dropped seeds like "Medulloblastoma" (no suffix in name) and
    "Echinococcus granulosus" (organism, not (disorder)).
    """
    if not seed_disease_names:
        return []

    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    _sys.path.insert(0, str(Path(__file__).resolve().parent / "dkr_policy"))
    from lexical_lookup import lookup_by_role  # noqa: E402

    candidate_roles = ("disease", "organism", "finding", "drug", "procedure")
    out: list[dict] = []
    seen: set[str] = set()
    unmatched: list[str] = []
    for seed in seed_disease_names:
        hits = lookup_by_role(seed, allowed_roles=candidate_roles,
                                max_results=top_k_per_seed)
        if not hits:
            unmatched.append(seed)
            continue
        for cui, _grp, term, _role in hits:
            if cui in seen:
                continue
            seen.add(cui)
            out.append({
                "cui": cui, "name": term, "cosine": 1.0,
                "seed": seed, "src": "lexical",
            })

    if unmatched:
        idx, enc = _load_umls_emb()
        # Build/cache candidate-role subset (disease/organism for L channel)
        global _CANDIDATE_ROW_MASK, _CANDIDATE_MATRIX, _CANDIDATE_CUIS, _CANDIDATE_NAMES
        if "_CANDIDATE_ROW_MASK" not in globals() or _CANDIDATE_ROW_MASK is None:
            roles = idx.get("role")
            if roles is None:
                # Legacy 61K openai matrix — fall back to (disorder) substring
                mask = np.array(["(disorder)" in n for n in idx["names"]])
            else:
                mask = np.array([r in candidate_roles for r in roles])
            _CANDIDATE_ROW_MASK = mask
            _CANDIDATE_MATRIX = idx["matrix"][mask]
            _CANDIDATE_CUIS = [c for c, m in zip(idx["cuis"], mask) if m]
            _CANDIDATE_NAMES = [n for n, m in zip(idx["names"], mask) if m]
        embs = enc.encode(unmatched)
        embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True),
                                   1e-12)
        for s_idx, seed in enumerate(unmatched):
            cos = _CANDIDATE_MATRIX @ embs[s_idx]
            n_added = 0
            for i in np.argsort(-cos):
                cui = _CANDIDATE_CUIS[i]
                if cui in seen:
                    continue
                seen.add(cui)
                out.append({
                    "cui": cui, "name": _CANDIDATE_NAMES[i],
                    "cosine": float(cos[i]), "seed": seed, "src": "sapbert",
                })
                n_added += 1
                if n_added >= top_k_per_seed:
                    break
    return out


# ── Smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Smoke test: cough + chest tightness at three durations\n")
    for t in [3, 30, 365]:
        print(f"\n{'='*70}")
        print(f"  t_obs = {t} days")
        print(f"{'='*70}")
        out = retrieve(
            ["cough", "chest tightness", "wheezing"], t_obs=t,
            patient_role="chief_complaint",
            k_sem=20, k_temp=20, k_anchor=20, k_final=15,
        )
        print(f"channels: {out['channel_stats']}")
        print(f"states:   {out['state_distribution']}")
        print(f"actions:  {out['action_distribution']}")
        print(f"\nTop-15 candidates:")
        for i, c in enumerate(out["candidates"]):
            srcs = "".join(sorted(c["sources"]))
            print(f"  {i+1:>2}. [{srcs:<3}] {c['name'][:45]:<45}  "
                  f"q={str(c['q']):<6}  compat={c['compat']:.2f}  "
                  f"state={c['state']:<12}  action={c['action']}")