"""On-demand BC: never silently return bc=0 for a disease/organism just because
its duration is uncached.

bc_for_cui (bc_llm_direct) returns 0.0 in two very different situations:
  1. node-type: finding/anatomy hubs have no clinical course  -> 0 is CORRECT
  2. cache-coverage: a disease/organism we simply never generated a duration for
     -> 0 is an ARTIFACT (fixable)

This wrapper distinguishes the two. On a cache miss for a disease/organism/finding
node it calls the LLM (role-conditional prompt, same as gen_durcrit_option_durations)
to generate {role,min_days,max_days}, appends to
cache/per_disease_durations_on_demand.jsonl, memoizes, and computes BC. Anatomy /
other roles still return 0. Empty results are also persisted so we don't re-query
a genuinely-no-duration concept on later runs.

Returns (bc, source_tag) where source_tag in
{invalid_t, role_no_duration, cache, durcrit, ondemand, gen_empty, gen_failed}.
"""
import json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent          # .../new_duration_spectrum
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "dkr_policy"))

import bc_llm_direct
from bc_llm_direct import bc_two_gaussians
from llm_client import call_llm

CACHE = ROOT / "cache"
ONDEMAND = CACHE / "per_disease_durations_on_demand.jsonl"
DURCRIT = CACHE / "per_disease_durations_durcrit_options.jsonl"

SIGMA_P = 0.30
GEN_MODEL = "gpt-5.4-mini"
DUR_ROLES = ("disease", "organism", "finding")

# role-conditional prompts (mirrors gen_durcrit_option_durations.ROLE_PROMPTS)
ROLE_PROMPTS = {
    "disease": """Provide the typical patient-presentation duration for the disease/disorder "{c}".
IMPORTANT: pick the axis that DEFINES this disease's time course:
 - acute_episode: self-limited acute illness (days)
 - subacute: weeks-scale course
 - chronic_baseline: chronic/progressive disease defined by months-to-years course (e.g. persistent depressive disorder >=2yr, chronic bronchitis, emphysema, slow-growing tumors)
 - lifelong: lifelong condition
Return JSON only:
{{"concept":"{c}","mentions":[{{"role":"acute_episode|subacute|chronic_baseline|lifelong","min_days":<num>,"max_days":<num>,"rationale":"<one line>"}}]}}
Use the course that a clinician would use to DISTINGUISH this disease from its differential. Median +-1 SD, not rare tail.""",
    "finding": """Provide the typical PERSISTENCE duration for the clinical finding "{c}" (how long it usually lasts).
If the name has a temporal qualifier (e.g. "Chronic ...", "Acute ..."), anchor on that.
Return JSON only:
{{"concept":"{c}","mentions":[{{"role":"acute|subacute|chronic","min_days":<num>,"max_days":<num>,"rationale":"<one line>"}}]}}
If no inherent time information, return "mentions":[].""",
    "organism": """Provide the typical incubation period to first symptoms for the infectious organism "{c}".
Return JSON only:
{{"concept":"{c}","mentions":[{{"role":"incubation_to_symptoms","min_days":<num>,"max_days":<num>,"rationale":"<one line>"}}]}}
From exposure to first clinical manifestation. Examples: E.coli ~1-3d, M.tuberculosis ~14-84d, Plasmodium ~7-30d.
If a taxonomic group with no defined incubation, return "mentions":[].""",
}


def _fit(mentions):
    """(mu, sigma) on log-time axis from the widest-range mention, or None."""
    ms = [m for m in (mentions or []) if m.get("min_days") and m.get("max_days")]
    if not ms:
        return None
    m = max(ms, key=lambda x: x["max_days"] - x["min_days"])
    lo, hi = float(m["min_days"]), float(m["max_days"])
    if lo <= 0 or hi < lo:
        return None
    mu = (math.log(lo) + math.log(hi)) / 2
    sigma = max((math.log(hi) - math.log(lo)) / (2 * 1.96), 0.10)
    return (mu, sigma)


def _load_durcrit():
    d = {}
    if DURCRIT.exists():
        for line in open(DURCRIT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = _fit(r.get("mentions"))
            if p:
                d[r["cui"]] = p
    return d


def _load_attempted():
    """Every CUI we have EVER generated for (mentions or empty), so we never re-query a concept
    the LLM already judged. Without this, non-temporal disease/finding/organism concepts (empty
    mentions) are skipped by _load_durations and thus re-generated on every single run — 72% of
    the on-demand traffic was exactly this. Persisting-but-not-reloading was the bug."""
    s = set()
    if ONDEMAND.exists():
        for line in open(ONDEMAND):
            try:
                s.add(json.loads(line).get("cui"))
            except json.JSONDecodeError:
                continue
    s.discard(None)
    return s


_durcrit = _load_durcrit()
_attempted = _load_attempted()   # CUIs already generated in a prior run (skip re-generation)
_memo = {}        # cui -> (mu,sigma) or None  (session memo, incl. generated + known-empty)
_gen_count = 0    # number of LLM generation calls this session


def _params(cui):
    """(mu,sigma) on log axis from any known source, or None. Does NOT generate."""
    if cui in _memo:
        return _memo[cui]
    p = bc_llm_direct._gaussian_params_for_cui(cui)   # on_demand+presentation+acute+llm+bcfix
    if p is not None:
        r = (p[0], p[1])
    elif cui in _durcrit:
        r = _durcrit[cui]
    else:
        r = None
    _memo[cui] = r
    return r


def _generate(cui, name, role):
    """Call LLM for a role-conditional duration; persist + memoize. Returns (mu,sigma) or None."""
    global _gen_count
    prompt = ROLE_PROMPTS[role].format(c=name)
    mentions = []
    status = "gen_empty"
    try:
        raw = call_llm(prompt, model=GEN_MODEL)
        _gen_count += 1
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            obj = json.loads(m.group(0))
            for ment in obj.get("mentions", []):
                try:
                    lo = float(ment["min_days"]); hi = float(ment["max_days"])
                    if lo > 0 and hi >= lo:
                        mentions.append({"role": ment.get("role", "unknown"),
                                         "min_days": lo, "max_days": hi,
                                         "source_phrase": ment.get("rationale", "")})
                except (KeyError, ValueError, TypeError):
                    continue
    except Exception:
        status = "gen_failed"
    # persist (even empty, so we never re-query) -- schema matches on_demand.jsonl
    rec = {"cui": cui, "disease": name, "role_class": role,
           "mentions": mentions, "n_mentions": len(mentions),
           "source": "ondemand_traversal", "model": GEN_MODEL}
    with open(ONDEMAND, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    p = _fit(mentions)
    _memo[cui] = p
    return p, ("ondemand" if p else status)


def bc_ondemand(cui, name, t_days, role, generate=True):
    """BC for a candidate at patient t_days, generating its duration on a cache miss.
    Returns (bc, source_tag)."""
    if t_days is None or t_days <= 0:
        return 0.0, "invalid_t"
    if role not in DUR_ROLES:
        return 0.0, "role_no_duration"
    p = _params(cui)
    src = "durcrit" if (p is not None and cui in _durcrit
                        and bc_llm_direct._gaussian_params_for_cui(cui) is None) else "cache"
    # Only generate for a CUI we have NEVER attempted. If it was attempted before and _params is
    # still None, the LLM already judged it non-temporal — reuse that verdict (bc=0), don't re-query.
    if p is None and generate and cui not in _attempted:
        p, src = _generate(cui, name, role)
        _attempted.add(cui)
    if p is None:
        return 0.0, "known_empty" if cui in _attempted else ("gen_empty" if generate else src)
    mu_d, sigma_d = p
    from bc_llm_direct import temporal_score   # honors WALKER_BC_MODE (overlap | interval_sample)
    return temporal_score(math.log(t_days), SIGMA_P, mu_d, sigma_d, key=cui), src


def gen_count():
    return _gen_count


if __name__ == "__main__":
    # smoke test: a deep gold known to be cached, and a likely-uncached organism bridge
    tests = [
        ("C0006034", "Borrelia burgdorferi", 14, "organism"),     # in durcrit
        ("C0027859", "Acoustic neuroma", 180, "disease"),         # in durcrit
        ("C0460002", "Structure of nervous system", 30, "anatomy"),  # anatomy -> 0
    ]
    for cui, name, t, role in tests:
        bc, src = bc_ondemand(cui, name, t, role)
        print(f"{name:34s} role={role:9s} t={t:>4}d  bc={bc:.3f}  src={src}")
    print("generation calls:", gen_count())
