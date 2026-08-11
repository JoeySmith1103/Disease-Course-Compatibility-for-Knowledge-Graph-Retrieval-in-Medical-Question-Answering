"""Lexical phrase → CUI lookup against cui_with_terms.csv.

Use case: F-channel seed entry. We want the CUI whose canonical surface
form is the symptom phrase, not the SapBERT-cosine-nearest concept of
all 394K. SapBERT is used downstream for disease scoring, not entry.

Build once (pickled), reuse on every retrieve() call.

Lookup strategies, in order:
  1. Exact (case-insensitive) — phrase matches a term verbatim
  2. Strip common suffixes ("(finding)", "(disorder)", etc) and re-try exact
  3. Token-overlap — phrase tokens ⊆ term tokens (or vice versa, with min 2 token overlap)
"""
from __future__ import annotations
import csv
import pickle
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TERMS_CSV = PROJECT_ROOT / "cui_with_terms.csv"
CACHE_PKL = PROJECT_ROOT / "cache" / "cui_terms_lookup.pkl"

_SUFFIXES = (
    " (disorder)", " (finding)", " (organism)", " (procedure)",
    " (substance)", " (qualifier value)", " (observable entity)",
    " (body structure)", " (morphologic abnormality)", " (disease)",
    " (syndrome)", " (product)", " (cell)", " (cell structure)",
    " (tissue)", " (event)", " (situation)", " (context-dependent category)",
)
_RETIRED = re.compile(r"\s*-\s*RETIRED\s*-\s*$", re.IGNORECASE)
# ICD-style classification prefix markers in SNOMED canonical names:
# "[D]Headache (context-dependent category)" → "headache"
_PREFIX_MARKER = re.compile(r"^\s*\[(d|x|v|q|m|so|exp)\]\s*", re.IGNORECASE)


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = _RETIRED.sub("", s)
    s = _PREFIX_MARKER.sub("", s)
    for sfx in _SUFFIXES:
        if s.endswith(sfx):
            s = s[: -len(sfx)]
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


_STOP_TOKENS = {"of", "and", "the", "in", "to", "with", "for", "an", "a", "or",
                 "by", "due", "as", "at", "on", "from"}


def _tokens_for_overlap(s: str) -> set[str]:
    """Extract content tokens for overlap matching: lowercase, hyphens split,
    drop stopwords + ≤2-char fragments. So "Methotrexate-induced pneumonitis"
    → {methotrexate, induced, pneumonitis} bridges to "Drug-induced pulmonary
    disease" → {drug, induced, pulmonary, disease} via {induced}."""
    tokens = re.split(r"[\s\-/]+", _normalize(s))
    return {t for t in tokens
            if t and len(t) > 2 and t not in _STOP_TOKENS}


_LOOKUP: dict | None = None


def _build() -> dict:
    """Build {normalized_term -> [(CUI, Group, original_term)]} plus an
    inverted CONTENT-token → term-keys index for O(|tokens|) token-overlap
    search. Content tokens split on whitespace AND hyphens so
    "drug-induced" tokenises to {drug, induced}."""
    by_term: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    token_index: dict[str, set[str]] = defaultdict(set)
    n = 0
    with open(TERMS_CSV) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 3:
                continue
            cui, group, term = row[0], row[1], row[2]
            if not term:
                continue
            norm = _normalize(term)
            if not norm:
                continue
            tup = (cui, group, term)
            if tup not in by_term[norm]:
                by_term[norm].append(tup)
            for tok in _tokens_for_overlap(term):
                token_index[tok].add(norm)
            n += 1
    return {"by_term": dict(by_term),
            "token_index": {k: list(v) for k, v in token_index.items()},
            "n_rows": n}


def _load() -> dict:
    global _LOOKUP
    if _LOOKUP is not None:
        return _LOOKUP
    if CACHE_PKL.exists():
        with open(CACHE_PKL, "rb") as f:
            _LOOKUP = pickle.load(f)
        return _LOOKUP
    print(f"[lexical_lookup] building from {TERMS_CSV}...")
    _LOOKUP = _build()
    CACHE_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PKL, "wb") as f:
        pickle.dump(_LOOKUP, f)
    print(f"[lexical_lookup] cached: {len(_LOOKUP['by_term']):,} unique terms "
          f"({_LOOKUP['n_rows']:,} total rows)")
    return _LOOKUP


def lookup_exact(phrase: str) -> list[tuple[str, str, str]]:
    """Return [(CUI, Group, original_term), ...] for an exact (normalized) match."""
    return _load()["by_term"].get(_normalize(phrase), [])


def lookup_token_overlap(phrase: str, min_tokens: int = 2,
                           max_results: int = 30,
                           min_jaccard: float = 0.0) -> list[tuple[str, str, str]]:
    """Return matches ranked by Jaccard token overlap (content tokens, with
    hyphen-split + stopword removal). Uses inverted index for speed.

    Adaptive min_tokens: queries with ≤2 content tokens (drug/procedure names
    like "prochlorperazine" or "Scrotal ultrasonography") relax to
    min_tokens=1 so they can match longer CSV terms ("Oral prochlorperazine",
    "Ultrasonography of total body"). Jaccard threshold filters noise.
    """
    data = _load()
    by_term = data["by_term"]
    tok_idx = data.get("token_index", {})
    p_tokens = _tokens_for_overlap(phrase)
    if not p_tokens:
        return []
    # Adaptive thresholds for short queries
    effective_min_tokens = min_tokens
    effective_min_jaccard = min_jaccard
    if len(p_tokens) <= 2:
        effective_min_tokens = 1
        # Higher Jaccard floor to avoid pulling random terms that share one
        # common token. 0.25 ≈ "1 in 4" overlap (e.g., 1-of-2 vs 1-of-3).
        effective_min_jaccard = max(min_jaccard, 0.25)
    if len(p_tokens) < effective_min_tokens:
        return []
    candidates: set[str] = set()
    for tok in p_tokens:
        candidates.update(tok_idx.get(tok, ()))
    scored: list[tuple[float, str]] = []
    for term in candidates:
        t_tokens = _tokens_for_overlap(term)
        inter = p_tokens & t_tokens
        if len(inter) < effective_min_tokens:
            continue
        jacc = len(inter) / max(len(p_tokens | t_tokens), 1)
        if jacc < effective_min_jaccard:
            continue
        scored.append((jacc, term))
    scored.sort(reverse=True)
    out: list[tuple[str, str, str]] = []
    for _jacc, term in scored:
        out.extend(by_term.get(term, ()))
        if len(out) >= max_results:
            break
    return out[:max_results]


def lookup(
    phrase: str,
    allowed_groups: tuple[str, ...] | None = None,
    fall_back_to_overlap: bool = True,
    max_results: int = 30,
) -> list[tuple[str, str, str]]:
    """Combined lookup with optional Group filter.
    Returns deduped (cui, group, term)."""
    hits = lookup_exact(phrase)
    if not hits and fall_back_to_overlap:
        hits = lookup_token_overlap(phrase, max_results=max_results)
    if allowed_groups:
        hits = [h for h in hits if h[1] in allowed_groups]
    # Dedup by CUI keeping first (longest matched term tends to come first
    # because we walked CSV in CSV order, but enforce uniqueness anyway).
    seen = set()
    uniq = []
    for h in hits:
        if h[0] in seen:
            continue
        seen.add(h[0])
        uniq.append(h)
    return uniq[:max_results]


# --- SapBERT semantic seed resolver --------------------------------------
# Token-overlap fails when canonical SNOMED label differs from common name
# (e.g., "Lyme disease" → SNOMED "Infection by Borrelia burgdorferi"; token
# overlap returns junk like "Suspected Lyme disease"). SapBERT cos on the
# 394K UMLS embedding matrix gives clean semantic resolution.
_SAPBERT_IDX = None
_SAPBERT_ENC = None


def _load_sapbert():
    global _SAPBERT_IDX, _SAPBERT_ENC
    if _SAPBERT_IDX is None:
        import os, pickle
        path = os.environ.get(
            "SAPBERT_PKL",
            str(PROJECT_ROOT / "cache" / "umls_broad_embeddings_sapbert.pkl"),
        )
        with open(path, "rb") as f:
            _SAPBERT_IDX = pickle.load(f)
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from duration_kg_rag import make_encoder
        _SAPBERT_ENC = make_encoder(_SAPBERT_IDX["encoder"])
    return _SAPBERT_IDX, _SAPBERT_ENC


_DISORDER_BOOST = None


def _get_disorder_boost(names):
    """Precompute +0.03 mask for (disorder)-suffixed names. ~5 ms one-time
    vs ~150 ms/call iterating 394K names per lookup."""
    global _DISORDER_BOOST
    if _DISORDER_BOOST is None:
        import numpy as np
        _DISORDER_BOOST = np.array(
            [0.03 if isinstance(n, str) and n.lower().endswith("(disorder)") else 0.0
             for n in names], dtype=np.float32)
    return _DISORDER_BOOST


def lookup_sapbert(
    phrase: str,
    top_k: int = 3,
    min_cos: float = 0.70,
    prefer_disorder: bool = True,
) -> list[tuple[str, str, str, float]]:
    """Semantic seed lookup via SapBERT cos against UMLS broad index.
    Returns [(cui, group, name, cos), ...] for top-k matches >= min_cos.
    `prefer_disorder` adds +0.03 to (disorder)-suffix concepts so canonical
    disease forms win ties."""
    import numpy as np
    idx, enc = _load_sapbert()
    matrix = idx["matrix"]; cuis = idx["cuis"]; names = idx["names"]
    q = enc.encode([phrase])
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    cos = matrix @ q[0]
    if prefer_disorder:
        cos = cos + _get_disorder_boost(names)
    top = np.argsort(-cos)[: top_k * 3]
    out = []
    for i in top:
        c = float(cos[i])
        if c < min_cos:
            break
        out.append((cuis[i], "Disorders", names[i], c))
        if len(out) >= top_k:
            break
    return out


# Convenience: filter by tui_roles role
def lookup_by_role(
    phrase: str,
    allowed_roles: tuple[str, ...] = ("finding",),
    max_results: int = 30,
) -> list[tuple[str, str, str, str]]:
    """Returns [(CUI, Group, term, role), ...] filtered by tui_roles role.
    Calls into tui_roles.get_role for each candidate.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dkr_policy"))
    from tui_roles import get_role  # noqa: E402

    hits = lookup(phrase, max_results=max_results * 2)
    out = []
    for cui, group, term in hits:
        role = get_role(cui)
        if role in allowed_roles:
            out.append((cui, group, term, role))
            if len(out) >= max_results:
                break
    return out


def main():
    """Smoke test."""
    # Force rebuild
    if CACHE_PKL.exists():
        print(f"removing existing cache {CACHE_PKL}")
        CACHE_PKL.unlink()
    _load()
    print()
    for q in ["abdominal pain", "shortness of breath", "rectal bleeding",
              "headache", "fever and chills", "echinococcus granulosus"]:
        hits = lookup_by_role(q, allowed_roles=("finding", "disease"))
        print(f"{q!r}: {len(hits)} hits")
        for h in hits[:3]:
            print(f"  {h[0]:<10} role={h[3]:<8} term={h[2][:60]!r}")


if __name__ == "__main__":
    main()
