"""Direct BC computation from LLM-judged disease durations (no MDN).

For each candidate disease (CUI), look up its LLM-derived [min_days, max_days]
in cache/per_disease_durations_llm.jsonl, fit a log-normal, then compute BC
against the patient prior 𝒩(log t_obs, σ_p).

Two log-normals:
  patient: 𝒩(log t_obs, σ_p²)
  disease: 𝒩(μ_d, σ_d²)  with
           μ_d = (log(min) + log(max)) / 2
           σ_d = (log(max) - log(min)) / (2 · 1.96)   # 95% CI width → σ

BC closed form for two Gaussians:
  BC = sqrt(2 σ_p σ_d / (σ_p² + σ_d²)) · exp(-(μ_p - μ_d)² / (4(σ_p² + σ_d²)))
"""
import json, math, os
from pathlib import Path
from functools import lru_cache


_DUR_CACHE = None  # cui -> (min_days, max_days, role)


def _load_durations():
    """Load LLM-judged durations from one or more sources. Prefer 'presentation'
    (acute-focused) when available; fall back to general LLM cache."""
    global _DUR_CACHE
    if _DUR_CACHE is not None: return _DUR_CACHE
    out = {}
    # Multi-source priority: bcfix_override (manually-corrected, 2026-05-13) >
    # on_demand (LLM-generated for walker-encountered CUIs, 2026-05-20) >
    # presentation (acute-focused) > acute > general LLM.
    # Override entries fix wide/wrong ranges identified in BC=0 rescue audit
    # (aud_070 Peripheral facial palsy was 42-180d in presentation.jsonl; the
    # override corrects to 14-42d based on Bell-palsy typical course).
    # Anchor to pipeline/cache (absolute), NOT cwd-relative 'cache/'. The writer bc_ondemand
    # appends on-demand durations to pipeline/cache; reading them cwd-relative pointed at a
    # DIFFERENT, stale file (new_duration_spectrum/cache), so every build re-generated every
    # duration via the LLM — 86% of the on_demand file was duplicate regenerations. Same dir
    # for both = generated durations are actually reused.
    _CACHE = Path(__file__).resolve().parent.parent.parent / "cache"   # pipeline/cache
    sources = [str(_CACHE / n) for n in (
        'per_disease_durations_bcfix_override.jsonl',
        'per_disease_durations_on_demand.jsonl',
        'per_disease_durations_presentation.jsonl',
        'per_disease_durations_acute.jsonl',
        'per_disease_durations_llm.jsonl',
    )]
    for src in sources:
        if not Path(src).exists(): continue
        for line in open(src):
            r = json.loads(line)
            cui = r.get('cui')
            if not cui or cui in out: continue
            mentions = r.get('mentions') or []
            if not mentions: continue
            # Pick the first acute_episode_duration if available, else any
            acute = [m for m in mentions if m.get('role') in
                     ('acute_episode_duration','symptomatic_course',
                      'onset_to_presentation','subacute_episode_duration')]
            chosen = acute[0] if acute else mentions[0]
            try:
                lo = float(chosen['min_days']); hi = float(chosen['max_days'])
                if lo > 0 and hi >= lo:
                    out[cui] = (lo, hi, chosen.get('role',''), src.split('/')[-1])
            except (KeyError, TypeError, ValueError):
                continue
    _DUR_CACHE = out
    return out


@lru_cache(maxsize=10000)
def _gaussian_params_for_cui(cui):
    """Return (μ_d, σ_d, source) on log-time axis, or None if unknown."""
    durs = _load_durations()
    if cui not in durs: return None
    lo, hi, role, src = durs[cui]
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    mu_d = (log_lo + log_hi) / 2
    # Treat [lo,hi] as 95% CI: σ = half-width / 1.96
    half_width = (log_hi - log_lo) / 2
    sigma_d = max(half_width / 1.96, 0.10)   # floor to prevent degenerate
    return (mu_d, sigma_d, src)


def bc_two_gaussians(mu_p, sigma_p, mu_d, sigma_d):
    """Bhattacharyya coefficient between two univariate Gaussians."""
    s2 = sigma_p**2 + sigma_d**2
    if s2 <= 0: return 0.0
    coef = math.sqrt(2 * sigma_p * sigma_d / s2)
    expo = -((mu_p - mu_d)**2) / (4 * s2)
    return coef * math.exp(expo)


def temporal_score(mu_p, sigma_p, mu_d, sigma_d, key: str = "") -> float:
    """Temporal-compatibility score between the patient log-normal 𝒩(mu_p,sigma_p) and the
    disease log-normal 𝒩(mu_d,sigma_d).

    Default (WALKER_BC_MODE=overlap): Bhattacharyya coefficient — overlap of the two *continuous
    distributions* (our method).

    WALKER_BC_MODE=interval_sample: INTERVAL/point baseline for the continuous-distribution ablation.
    Draw one sample from each distribution (by its probability) and score their closeness via
    inverse-distance weighting 1/(1+|Δ|) (Shepard 1968, IDW — a citable standard, not ad-hoc).
    Seeded per `key` (disease CUI) so it is deterministic/reproducible across runs.

    The seed goes through hashlib, NOT the built-in hash(): Python randomises string hashing per
    process (PYTHONHASHSEED, on by default since 3.3), so hash(("bc_is", cui, ...)) returns a
    different value every interpreter start. This baseline was therefore drawing fresh samples on
    every run while claiming to be reproducible — three separate processes scoring the same
    (patient, disease) pair returned 0.5482 / 0.8345 / 0.4627. Deterministic within a run, not
    across them, which is the failure mode that hides longest."""
    import os
    if os.environ.get("WALKER_BC_MODE", "overlap") == "interval_sample":
        import hashlib, random
        _k = f"bc_is|{key}|{round(mu_p, 3)}|{round(mu_d, 3)}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(_k).digest()[:8], "big"))
        x_p = rng.gauss(mu_p, sigma_p)
        x_d = rng.gauss(mu_d, sigma_d)
        return 1.0 / (1.0 + abs(x_p - x_d))
    return bc_two_gaussians(mu_p, sigma_p, mu_d, sigma_d)


def bc_for_cui(cui: str, t_days: float, sigma_p: float = 0.30) -> float:
    """Compute the temporal score for a candidate's LLM-derived duration vs patient prior at t_days.
    Returns 0.0 if disease not in LLM cache. Score mechanism set by WALKER_BC_MODE (see temporal_score).
    """
    if t_days is None or t_days <= 0: return 0.0
    params = _gaussian_params_for_cui(cui)
    if params is None: return 0.0
    mu_d, sigma_d, _ = params
    mu_p = math.log(t_days)
    return temporal_score(mu_p, sigma_p, mu_d, sigma_d, key=cui)


def coverage_stats():
    durs = _load_durations()
    return {'n_diseases': len(durs), 'sources_used': set(d[3] for d in durs.values())}


if __name__ == '__main__':
    # Quick self-test on perturbation pairs
    durs = _load_durations()
    print(f'Loaded {len(durs)} disease durations')

    PAIRS = [
        ('Acute viral gastroenteritis', 'Inflammatory bowel disease'),
        ('Septic arthritis',            'Osteoarthritis'),
        ('Migraine',                    'Tension-type headache'),
        ('Acute bronchitis',            'Chronic bronchitis'),
        ('Acute appendicitis',          'Irritable bowel syndrome'),
        ('Pulmonary embolism',          'COPD'),
        ('Hyperthyroidism',             'Diabetes mellitus type 2'),
        ('Influenza',                   'Tuberculosis'),
        ('Acute vestibular neuritis',   'Meniere disease'),
        ('Acute viral infection',       'Major depressive disorder'),
    ]

    # Find CUIs by name (substring match)
    name2cui = {}
    for cui, (lo, hi, role, src) in durs.items():
        # We don't have name in this cache, need to load original record
        pass
    # Reload and build name index
    import json
    name_index = {}
    for src in ['cache/per_disease_durations_bcfix_override.jsonl',
                'cache/per_disease_durations_presentation.jsonl',
                'cache/per_disease_durations_acute.jsonl',
                'cache/per_disease_durations_llm.jsonl']:
        if not Path(src).exists(): continue
        for line in open(src):
            r = json.loads(line)
            disease = (r.get('disease','') or '').lower().strip()
            cui = r.get('cui')
            if disease and cui and disease not in name_index:
                name_index[disease] = cui
    print(f'Name index: {len(name_index)} diseases')

    def find_cui(query):
        q = query.lower()
        for name, cui in name_index.items():
            if q in name or name in q:
                return cui, name
        return None, None

    print()
    print(f'{"Pair @ t":48s} | acute BC | chronic BC | winner')
    print('-' * 82)
    T_GRID = [0.5, 7.0, 90.0, 1825.0]
    n_correct_at_short = 0
    n_correct_at_long = 0
    n_pairs = 0
    for acute, chronic in PAIRS:
        cui_a, na = find_cui(acute)
        cui_c, nc = find_cui(chronic)
        if not cui_a or not cui_c:
            print(f'  {acute} or {chronic}: not found in cache (cui_a={cui_a}, cui_c={cui_c})')
            continue
        n_pairs += 1
        for t in T_GRID:
            bc_a = bc_for_cui(cui_a, t)
            bc_c = bc_for_cui(cui_c, t)
            winner = 'A' if bc_a > bc_c else 'C'
            print(f'  {acute[:18]:>18} vs {chronic[:18]:<18} @ {t:>6g}d | {bc_a:.3f}   | {bc_c:.3f}    | {winner}')
            if t == 0.5 and winner == 'A': n_correct_at_short += 1
            if t == 1825 and winner == 'C': n_correct_at_long += 1
        print()
    if n_pairs:
        print(f'\nDirection at t=0.5d (acute should win): {n_correct_at_short}/{n_pairs}')
        print(f'Direction at t=1825d (chronic should win): {n_correct_at_long}/{n_pairs}')
