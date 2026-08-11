"""UMLS Semantic Type-based role assignment for DKR-Policy.

Replaces the legacy "(disorder)" string filter with proper TUI-based
typing. Each CUI may have multiple TUIs; we assign a primary role by
priority:

    disease  >  finding  >  anatomy  >  drug  >  procedure  >  organism  >  other

This priority ensures that, e.g., a CUI tagged as both T184 (Sign or
Symptom) and T047 (Disease or Syndrome) is treated as a disease (the
strongest signal) and enters the candidate pool — rather than being
demoted to "finding" by the lexical sort order.

Roles drive two decisions in v10:
  1. Candidate pool admission: only role == "disease" enters BC scoring
  2. Evidence narrative labelling: mediators in KG paths are labelled
     by their role so the LLM sees "via finding X" / "via anatomy Y"
"""
from __future__ import annotations
import csv
import json
import ast
from pathlib import Path
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CUI2TUI_CSV  = PROJECT_ROOT / "cui_to_tui.csv"
TUI_INFO_JSON = PROJECT_ROOT / "tui_info.json"


# ── TUI sets per role ───────────────────────────────────────────────
# DISO group minus findings/symptoms
DISEASE_TUIS = frozenset({
    "T020",  # Acquired Abnormality
    "T190",  # Anatomical Abnormality
    "T049",  # Cell or Molecular Dysfunction
    "T019",  # Congenital Abnormality
    "T047",  # Disease or Syndrome
    "T050",  # Experimental Model of Disease
    "T037",  # Injury or Poisoning
    "T048",  # Mental or Behavioral Dysfunction
    "T191",  # Neoplastic Process
    "T046",  # Pathologic Function
})

# Findings/symptoms — entry points for retrieval, NOT candidates
FINDING_TUIS = frozenset({
    "T033",  # Finding
    "T184",  # Sign or Symptom
    "T034",  # Laboratory or Test Result
    "T201",  # Clinical Attribute
})

# Anatomical anchors
ANATOMY_TUIS = frozenset({
    "T017", "T029", "T023", "T030", "T031", "T022",
    "T025", "T026", "T018", "T021", "T024",
})

# Drugs / chemicals — excluded from candidate pool
DRUG_TUIS = frozenset({
    "T121",  # Pharmacologic Substance
    "T200",  # Clinical Drug
    "T195",  # Antibiotic
    "T125",  # Hormone
    "T127",  # Vitamin
    "T129",  # Immunologic Factor
    "T131",  # Hazardous or Poisonous Substance
    "T123",  # Biologically Active Substance
    "T122",  # Biomedical or Dental Material
    "T103",  # Chemical (general)
    "T109",  # Organic Chemical
    "T197",  # Inorganic Chemical
    "T120", "T104", "T196", "T126", "T114", "T192", "T116",
})

# Procedures — excluded from candidate pool
PROCEDURE_TUIS = frozenset({
    "T060",  # Diagnostic Procedure
    "T061",  # Therapeutic or Preventive Procedure
    "T058",  # Health Care Activity
    "T059",  # Laboratory Procedure
    "T063", "T062", "T065",
})

# Organisms (pathogens, vectors) — mediators of etiology
ORGANISM_TUIS = frozenset({
    "T005",  # Virus
    "T007",  # Bacterium
    "T004",  # Fungus
    "T194",  # Archaeon
    "T204",  # Eukaryote
    "T001",  # Organism
    "T008", "T010", "T015", "T011", "T012", "T013", "T014",
})

# Physiologic processes — used as morphology/mechanism anchors
PROCESS_TUIS = frozenset({
    "T038",  # Biologic Function
    "T039", "T040", "T041", "T042", "T043", "T044", "T045",
    "T067", "T070", "T068", "T069",
})

# Body substances / cell components — anchors
SUBSTANCE_TUIS = frozenset({
    "T031",  # Body Substance (already in ANAT but listed for clarity)
    "T167",  # Substance (general)
})

# Role priority — assigned to the highest-priority role a CUI matches
ROLE_PRIORITY = [
    ("disease",   DISEASE_TUIS),
    ("finding",   FINDING_TUIS),
    ("anatomy",   ANATOMY_TUIS),
    ("drug",      DRUG_TUIS),
    ("procedure", PROCEDURE_TUIS),
    ("organism",  ORGANISM_TUIS),
    ("process",   PROCESS_TUIS),
    ("substance", SUBSTANCE_TUIS),
]


# ── Lazy CUI → TUIs lookup ──────────────────────────────────────────
_CUI2TUI: dict[str, list[str]] | None = None


def _load_cui2tui() -> dict[str, list[str]]:
    global _CUI2TUI
    if _CUI2TUI is not None:
        return _CUI2TUI
    out: dict[str, list[str]] = {}
    with open(CUI2TUI_CSV) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 2:
                continue
            cui, tui_list_str = row[0], row[1]
            try:
                tuis = ast.literal_eval(tui_list_str)
                out[cui] = list(tuis)
            except (ValueError, SyntaxError):
                continue
    _CUI2TUI = out
    return out


@lru_cache(maxsize=200_000)
def get_tuis(cui: str) -> tuple[str, ...]:
    """Return the TUI list for a CUI, or empty tuple if unknown."""
    return tuple(_load_cui2tui().get(cui, []))


@lru_cache(maxsize=200_000)
def get_role(cui: str) -> str:
    """Return the primary role of a CUI based on TUI priority.

    Returns one of: 'disease', 'finding', 'anatomy', 'drug',
    'procedure', 'organism', 'process', 'substance', 'other'.
    Returns 'unknown' if CUI not in cui_to_tui.csv.
    """
    tuis = get_tuis(cui)
    if not tuis:
        return "unknown"
    tui_set = set(tuis)
    for role, role_tuis in ROLE_PRIORITY:
        if tui_set & role_tuis:
            return role
    return "other"


# Default candidate-admission roles.
#
# Empirical motivation: a 60-item bench325 spot-check showed ~15% of
# gold MCQ options resolve to non-disease CUIs (organism for "causative
# pathogen?", finding for laboratory/sign answers, procedure for
# "best next step?", drug for treatment). The audit_subgraph_helpful
# result (+8.6pp conditional accuracy when gold is in the retrieved
# subgraph) attributes the gain to gold appearing in the LLM's evidence
# block. Disease-only filtering silently drops those non-disease golds
# at the pool-admission step, so even a perfect SapBERT retriever
# couldn't help — the CUI never reaches ranking.
#
# Anatomy / process stay excluded: they are mediators along the path
# (e.g. "Liver" linking hepatitis to its findings), not the gold MCQ
# concept. Admitting them dilutes ranking with topical context.
DEFAULT_CANDIDATE_ROLES: tuple[str, ...] = (
    "disease", "finding", "drug", "procedure", "organism",
)


def is_candidate(
    cui: str,
    allowed_roles: tuple[str, ...] = DEFAULT_CANDIDATE_ROLES,
) -> bool:
    """Whether a CUI is admissible as a candidate (gets BC scoring)."""
    return get_role(cui) in allowed_roles


def role_label_for_evidence(role: str) -> str:
    """Human-readable label for KG path narrative."""
    return {
        "disease":   "disease",
        "finding":   "finding",
        "anatomy":   "anatomical site",
        "drug":      "drug",
        "procedure": "procedure",
        "organism":  "pathogen",
        "process":   "biological process",
        "substance": "substance",
        "other":     "concept",
        "unknown":   "concept",
    }.get(role, "concept")


# ── Sanity check / smoke test ──────────────────────────────────────
def main():
    """Quick audit of role distribution."""
    cui2tui = _load_cui2tui()
    print(f"Loaded {len(cui2tui)} CUI → TUI mappings")
    role_counts: dict[str, int] = {}
    for cui in list(cui2tui.keys())[:50000]:  # sample
        role = get_role(cui)
        role_counts[role] = role_counts.get(role, 0) + 1
    print("\nRole distribution (first 50K CUIs):")
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        print(f"  {role:<12} {count:>6}")

    print("\nSpot checks:")
    # Pneumonia CUI should be disease
    test_cuis = ["C0032285", "C0032285", "C0010054", "C0024117"]
    for cui in test_cuis:
        tuis = get_tuis(cui)
        role = get_role(cui)
        print(f"  {cui}: TUIs={list(tuis)}  role={role}")


if __name__ == "__main__":
    main()
