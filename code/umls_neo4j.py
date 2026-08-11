"""Neo4j wrapper for the UMLS / SNOMED CT subset.

Schema (probed from server):
  Nodes:
    :Concept  (CUI: str, name: str)        # 443K canonical concepts
    :Atom     (AUI, CUI, name)              # 1.67M term variants

  Relationships (Concept-Concept, all directed):
    PAR (parent), CHD (child)               2.38M each — hierarchical (broader)
    RB  (broader), RN (narrower)            150K each   — hierarchical (related)
    RO  (other related)                     9.44M       — clinical relations
    SY  (synonym)                           128K        — same concept

Top RELA values on RO (clinical):
    has_finding_site / finding_site_of            804K each  ← anatomy ↔ disease
    has_associated_morphology / morph_of          605K       ← morphology ↔ disease
    method_of / has_method                        595K       ← procedure relations
    has_pathological_process / process_of         129K       ← mechanism ↔ disease
    has_causative_agent / causative_agent_of      183K       ← cause ↔ disease
    is_interpreted_by / interprets                199K       ← finding ↔ measurement

Disease-type filter: SNOMED CT names diseases with "(disorder)" suffix.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable, Optional

from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "admin")


_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600,
        )
    return _driver


@contextmanager
def session():
    s = get_driver().session()
    try:
        yield s
    finally:
        s.close()


# ── Entity linking ──────────────────────────────────────────────────────


SNOMED_DISORDER_SUFFIX = ("(disorder)", "(finding)", "(disease)", "(syndrome)")


def is_disorder_name(name: str) -> bool:
    """True if the SNOMED concept name has a disease-like suffix."""
    if not name:
        return False
    nm = name.lower()
    return any(s in nm for s in SNOMED_DISORDER_SUFFIX)


SNOMED_QUALIFIERS = (
    " (disorder)", " (finding)", " (disease)", " (syndrome)",
    " (morphologic abnormality)", " (procedure)", " (substance)",
    " (qualifier value)", " (observable entity)",
)


def lookup_concepts(
    phrase: str,
    limit: int = 8,
    prefer_disorder: bool = True,
) -> list[dict]:
    """Resolve a phrase to UMLS concepts. Tries exact match (with/without
    SNOMED qualifier suffix); then prefix matching ranked by name length
    (shortest = most canonical); falls back to CONTAINS only if needed.

    Returns [{cui, name, kind}]. kind ∈ {'exact', 'qualified', 'prefix',
    'contains'}.
    """
    if not phrase or not phrase.strip():
        return []
    phrase_lc = phrase.lower().strip()
    results = []
    seen_cuis = set()

    def _add(rows, kind):
        for r in rows:
            cui = r["cui"]; name = r["name"]
            if cui in seen_cuis:
                continue
            seen_cuis.add(cui)
            results.append({"cui": cui, "name": name, "kind": kind})

    with session() as s:
        # 1) exact bare-name match
        rows = list(s.run(
            "MATCH (c:Concept) WHERE toLower(c.name) = $p "
            "RETURN c.CUI AS cui, c.name AS name LIMIT 5",
            p=phrase_lc,
        ))
        _add(rows, "exact")

        # 2) exact match with SNOMED qualifier
        for q in SNOMED_QUALIFIERS:
            rows = list(s.run(
                "MATCH (c:Concept) WHERE toLower(c.name) = $p "
                "RETURN c.CUI AS cui, c.name AS name LIMIT 3",
                p=phrase_lc + q,
            ))
            _add(rows, "qualified")

        # IMPORTANT: if we have exact/qualified matches, STOP. Prefix is
        # noisy — it'd bring in compound concepts like "Fever associated
        # with AIDS" which pollute the search.
        if results:
            return results[:limit]

        # 3) prefix only when nothing exact found. Filter out compound names.
        rows = list(s.run(
            "MATCH (c:Concept) WHERE toLower(c.name) STARTS WITH $p "
            "RETURN c.CUI AS cui, c.name AS name "
            "ORDER BY size(c.name) ASC LIMIT 100",
            p=phrase_lc,
        ))
        # filter compounds: skip names with these noise indicators
        compound_markers = (" associated ", " with ", " - ", " due to ",
                            " in chronic ", " of unknown ", " co-occurrent ",
                            " caused by ", "[d]")
        clean_rows = [
            r for r in rows
            if not any(m in r["name"].lower() for m in compound_markers)
        ]
        if prefer_disorder:
            disorder_rows = [r for r in clean_rows if is_disorder_name(r["name"])]
            if disorder_rows:
                _add(disorder_rows[: limit], "prefix")
        if not results:
            _add(clean_rows[: limit], "prefix")

        # 4) CONTAINS only as final fallback
        if not results:
            rows = list(s.run(
                "MATCH (c:Concept) WHERE toLower(c.name) CONTAINS $p "
                "RETURN c.CUI AS cui, c.name AS name "
                "ORDER BY size(c.name) ASC LIMIT 30",
                p=phrase_lc,
            ))
            clean_rows = [
                r for r in rows
                if not any(m in r["name"].lower() for m in compound_markers)
            ]
            _add(clean_rows[: limit], "contains")

    return results[:limit]


# ── Neighborhood expansion ──────────────────────────────────────────────


# Relations useful for diagnostic reasoning (anchor → disease)
DIAGNOSTIC_INBOUND_RELAS = [
    "finding_site_of",          # anatomy → disease
    "associated_morphology_of",  # morphology → disease
    "pathological_process_of",  # mechanism → disease
    "has_causative_agent",      # disease ← organism (we want disease ← organism)
    "interprets",               # measurement → finding
    "may_be_a",                 # broader generalization
]

# When traversing FROM a disease to find related diseases:
DIAGNOSTIC_LATERAL_RELAS = [
    "associated_with",
    "due_to",
    "complication_of",
    "occurs_after",
    "manifestation_of",
    "may_be_finding_of_disease",
]


def expand_neighbors(
    cuis: Iterable[str],
    rel_types: Optional[list[str]] = None,
    rela_filter: Optional[list[str]] = None,
    limit: int = 500,
    direction: str = "both",  # 'out' | 'in' | 'both'
) -> list[dict]:
    """Return one-hop neighbors. Returns [{src_cui, dst_cui, dst_name, rel, rela}]."""
    cuis = list(cuis)
    if not cuis:
        return []
    rel_clause = "[r]" if rel_types is None else f"[r:{ '|'.join(rel_types) }]"
    arrow = {"out": "->", "in": "<-", "both": "-"}[direction]
    rela_clause = ""
    if rela_filter:
        rela_clause = "AND r.RELA IN $rela_filter "
    cypher = (
        f"MATCH (c:Concept)-{rel_clause}{arrow}(n:Concept) "
        f"WHERE c.CUI IN $cuis {rela_clause}"
        "RETURN c.CUI AS src_cui, n.CUI AS dst_cui, n.name AS dst_name, "
        "type(r) AS rel, r.RELA AS rela "
        "LIMIT $limit"
    )
    with session() as s:
        rows = s.run(cypher, cuis=cuis, rela_filter=rela_filter, limit=limit)
        return [dict(r) for r in rows]


def get_concept_names(cuis: Iterable[str]) -> dict[str, str]:
    """Resolve CUIs to canonical names."""
    cuis = list(cuis)
    if not cuis:
        return {}
    with session() as s:
        rows = s.run(
            "MATCH (c:Concept) WHERE c.CUI IN $cuis "
            "RETURN c.CUI AS cui, c.name AS name",
            cuis=cuis,
        )
        return {r["cui"]: r["name"] for r in rows}


# ── Smoke test ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Lookup 'Crohn disease':")
    for r in lookup_concepts("Crohn disease", limit=3):
        print(f"  [{r['kind']:<10}] {r['cui']}  {r['name']}")
    print()
    print("Lookup 'pneumonia' (disorder only):")
    for r in lookup_concepts("pneumonia", limit=5, only_disorder=True):
        print(f"  [{r['kind']:<10}] {r['cui']}  {r['name']}")
    print()
    print("Expand 'Crohn disease' candidate via finding_site:")
    cuis = [r["cui"] for r in lookup_concepts("Crohn disease", limit=2)]
    if cuis:
        for r in expand_neighbors(cuis, rela_filter=["has_finding_site"], limit=8)[:8]:
            print(f"  {r['rela']:<25} → {r['dst_name']}")
