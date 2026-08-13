#!/usr/bin/env python3
"""One-sided temporal compatibility, as a replacement for the Bhattacharyya overlap.

WHY. bc compares the patient's ELAPSED time against the disease's TOTAL COURSE as two log-normals
and scores their overlap. Those are not the same quantity. A patient sixty days into a disease that
runs for years is a completely ordinary presentation, but the two distributions barely overlap, so
bc scores it as incompatible and the walker demotes the right answer.

Measured on the gold candidates (329 / medbullets), grouped by the gold disease's own course:

    course ≤14d      gold bc 0.720 / 0.030     patient 2d / 0.2d
    course 14–90d    gold bc 0.535 / 0.909     patient 7d / 21d
    course >90d      gold bc 0.477 / 0.137     patient 60d / 21d      ← 305 and 65 candidates
                     distractor bc ≈ 0.65–0.67 throughout

Acute golds score above the distractor median; chronic golds — the majority — score far below it.
That is the whole of bc's negative separation.

WHAT THIS COMPUTES INSTEAD. P(course ≥ elapsed): the probability that the disease can still be
running after the time the patient has had symptoms. A ten-year disease is compatible with any
elapsed time; a disease that resolves in a week is not compatible with two months of symptoms.

    compat = 1 − Φ((log t_obs − μ_d) / σ_d)

This is deliberately one-sided and therefore weaker: it cannot separate two chronic diseases from
each other, because elapsed time genuinely does not. It is meant to stop bc from actively demoting
correct chronic answers, not to add discrimination that the signal does not carry.
"""
import json, math, os
from functools import lru_cache
from pathlib import Path

PIPE = Path(__file__).resolve().parent
_D = None


def _load():
    """cui -> (mu_d, sigma_d) of the disease's log-normal course, same fit bc_llm_direct uses."""
    global _D
    if _D is not None:
        return _D
    _D = {}
    for f in ("cache/per_disease_durations_llm.jsonl",
              "cache/per_disease_durations_on_demand.jsonl"):
        p = PIPE / f
        if not p.exists():
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ms = r.get("mentions") or []
            lo = [m.get("min_days") for m in ms if isinstance(m.get("min_days"), (int, float)) and m["min_days"] > 0]
            hi = [m.get("max_days") for m in ms if isinstance(m.get("max_days"), (int, float)) and m["max_days"] > 0]
            if not lo or not hi:
                continue
            mn, mx = min(lo), max(hi)
            if mx <= mn:
                mx = mn * 1.5
            mu = (math.log(mn) + math.log(mx)) / 2
            sd = max((math.log(mx) - math.log(mn)) / (2 * 1.96), 1e-3)
            _D[r["cui"]] = (mu, sd)
    return _D


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@lru_cache(maxsize=200000)
def compat_for_cui(cui: str, t_days: float) -> float:
    """P(disease course ≥ t_days). 0.0 when the disease has no stored duration."""
    if not t_days or t_days <= 0:
        return 0.0
    d = _load().get(cui)
    if d is None:
        return 0.0
    mu, sd = d
    return 1.0 - _phi((math.log(t_days) - mu) / sd)


if __name__ == "__main__":
    # sanity: a 60-day presentation should be compatible with a chronic disease and not with a
    # week-long one — the case bc gets backwards
    import sys
    D = _load()
    print(f"loaded {len(D)} disease duration fits")
    for cui, name in [("C0025286", "Meningioma"), ("C0001623", "Adrenal insufficiency"),
                      ("C0009450", "Common cold")]:
        if cui in D:
            mu, sd = D[cui]
            print(f"  {name:24} course≈{math.exp(mu):8.1f}d  "
                  + "  ".join(f"t={t}d→{compat_for_cui(cui, t):.2f}" for t in (2, 30, 60, 365)))
