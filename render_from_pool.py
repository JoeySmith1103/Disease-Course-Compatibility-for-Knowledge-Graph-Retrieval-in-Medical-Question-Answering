#!/usr/bin/env python3
"""Re-rank a stored pool under new parameters and freeze the resulting prompts.

This is the point of build_pools.py. A sweep over top-K, λ, μ or the utility formula becomes a
pass over pool/<ds>/<method>.jsonl — no Neo4j, no LLM, seconds instead of forty minutes — and the
output is a normal frozen file that run_reader.py consumes like any other.

WHAT IT CANNOT DO: τ (min_score), max_hops and neighbor_limit gated EXPANSION during the walk, so
candidates they excluded were never scored and are not in the pool. Asking for a lower τ here would
silently return the same pool under a new label, so it is refused rather than approximated.

The variant name encodes the parameters, so two sweeps cannot overwrite each other and a result
file can be traced back to the settings that produced it:
    frozen/<ds>/<method>__k20_l0.6_m0.08.json

Usage:
  DATASET=329 METHOD=walker TOP_K=20 LAMBDA=0.6 python3 pipeline/render_from_pool.py
  DATASET=329 METHOD=walker UTILITY='cos*bc' python3 pipeline/render_from_pool.py
  DATASET=329 METHOD=walker TOP_K=20 DRY=1 python3 pipeline/render_from_pool.py   # 只看不寫
"""
import bisect, json, os, re, statistics as st, importlib.util
from pathlib import Path

PIPE = Path(__file__).resolve().parent
DS      = os.environ.get("DATASET", "329")
METHOD  = os.environ.get("METHOD", "walker")
TOP_K   = int(os.environ.get("TOP_K", "10"))
LAMBDA  = float(os.environ.get("LAMBDA", "0.3"))
MU      = float(os.environ.get("MU", "0.08"))
# role quota mirrors format_kg_block's disease[:7] + non_dis[:3]; set 0 to rank purely by score
Q_DIS   = int(os.environ.get("QUOTA_DISEASE", "7"))
Q_OTHER = int(os.environ.get("QUOTA_OTHER", "3"))
UTILITY = os.environ.get("UTILITY", "")     # e.g. "cos*bc" or "cos + 0.3*bc - 0.08*hop"
# SAMPLE=random turns an unscored pool into an actual random control.
#
# raw_1hop is described as an unranked baseline, but it is not neutral: expand() walks the seeds in
# order and stops at the cap, and _collect_all_seeds puts the LLM's differential diagnoses first.
# Its top-10 is therefore mostly the neighbourhood of the model's own best guess — a strong prior
# wearing a baseline's name. Sampling uniformly from the retained 50 removes that ordering effect
# and asks what the 1-hop neighbourhood is worth on its own.
SAMPLE  = os.environ.get("SAMPLE", "")
MAX_HOP = os.environ.get("MAX_HOP")
MAX_HOP = int(MAX_HOP) if MAX_HOP not in (None, "") else None
TAG     = os.environ.get("VARIANT", "")
DRY     = os.environ.get("DRY", "")
TEMPLATE_FILE = os.environ.get("TEMPLATE_FILE", "")

_STOP = {"of", "the", "a", "an", "and", "or", "to", "in", "with", "disorder", "finding",
         "disease", "syndrome", "nos", "due", "by", "on", "structure", "product", "substance"}
_Q_STOP = _STOP | {"patient", "history", "year", "old", "man", "woman", "presents"}
_WORD = re.compile(r"[a-z0-9]+")


def name_tokens(name):
    return {w for w in _WORD.findall((name or "").lower()) if w not in _STOP and len(w) > 2}


def question_tokens(q):
    return {w for w in _WORD.findall((q or "").lower()) if w not in _Q_STOP and len(w) > 2}


def _overlap(a, b):
    i = len(a & b)
    return 0.0 if not i else 2 * i / (len(a) + len(b))


def novelty(c, q_toks):
    """1 − share of the candidate's name already present in the question.

    cos rewards resembling the vignette, and what resembles a vignette most is a reworded symptom,
    which carries no information the reader does not already have. Measured on these pools the gold
    rate falls monotonically as that overlap rises: 1.12% at zero overlap down to 0.27% above 0.66
    on 329 (0.50% → 0.18% on MedBullets).
    """
    t = name_tokens(c.get("name"))
    return 1.0 - len(t & q_toks) / max(len(t), 1)


def select_diverse(cands, score, top_k, delta, q_dis=7, q_other=3):
    """Greedy selection charging a candidate for repeating one already chosen.

    In the shipped top-10, 4.6 slots per question on 329 and 5.3 on MedBullets overlap another slot
    by at least 0.5 on name tokens — half the block restates the complaint. Weights cannot fix that:
    near-duplicates share nearly identical cos and move together whatever the coefficients are.

    delta=0 reproduces plain top-k, so this is a strict extension.
    """
    if delta <= 0:
        return None
    pool = sorted(cands, key=lambda c: -score(c))
    toks = {id(c): name_tokens(c.get("name")) for c in pool}
    picked, n_dis = [], 0
    while pool and len(picked) < top_k:
        best, best_v = None, None
        for c in pool:
            if c.get("role") == "disease" and n_dis >= q_dis and len(picked) < top_k - 1:
                continue
            pen = max((_overlap(toks[id(c)], toks[id(p)]) for p in picked), default=0.0)
            v = score(c) - delta * pen
            if best_v is None or v > best_v:
                best, best_v = c, v
        if best is None:
            break
        picked.append(best); pool.remove(best)
        if best.get("role") == "disease":
            n_dis += 1
    return picked


_spec = importlib.util.spec_from_file_location("P", PIPE / "prompts.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)

pool_path = PIPE / f"pool/{DS}/{METHOD}.jsonl"
if not pool_path.exists():
    raise SystemExit(f"沒有 {pool_path} — 先跑 python3 pipeline/build_pools.py --walk {DS}")
records = [json.loads(l) for l in open(pool_path)]
_pd1 = lambda v: (float(v[0]) if isinstance(v, (list, tuple)) and v
                  else (float(v) if isinstance(v, (int, float)) else None))
_PD = {r["uid"]: _pd1(r.get("patient_days")) for r in records}
scored = all("cos" in (r["candidates"][0] if r["candidates"] else {"cos": 1}) for r in records[:5])
if (LAMBDA != 0.3 or MU != 0.08 or UTILITY) and not scored:
    raise SystemExit(f"{METHOD} 的 pool 沒有 cos/bc/hop 分項，只能調 TOP_K，不能改 λ/μ/utility")


# ── utility variants ─────────────────────────────────────────────────────────
# MODE=fixed       score = cos + λ·bc − μ·hop                       (shipped)
# MODE=criticality score = (1−w)·cos + w·bc − μ·hop,  w = LLM's 0–1 judgement of how much THIS
#                  question's diagnosis turns on the time course
# MODE=adaptive    same convex form, but w is derived from cos itself: where the symptom match is
#                  already confident, let semantics lead; where it is weak, lean on time
#
# The two forms are NOT on the same scale. `fixed` adds an unnormalised λ·bc on top of a full-weight
# cos; the convex forms split one unit of weight between them. Since bc runs about 0.4 higher than
# cos in this pool (non-zero bc median ≈0.67 vs cos median ≈0.49), a convex w has more bite than the
# same-looking λ: w ≈ 0.23 reproduces the shipped λ=0.3, so w=0.5 is roughly twice today's temporal
# pull, not half. W_MAX caps that.
MODE   = os.environ.get("UTILITY_MODE", "fixed")
# BC_SRC=onesided swaps the stored Bhattacharyya overlap for P(course >= elapsed), recomputed per
# candidate from the same duration cache. The overlap penalises a patient who presents early in a
# long illness — the majority case — and that is the whole of bc's negative gold separation
# (329: -0.169 -> +0.132, medbullets: -0.415 -> +0.169). See bc_onesided.py.
# BC_SRC=onesided reads the precomputed `bc_onesided` field. It used to recompute from the
# disease-duration cache, which is 48 MB and not in the repo — so a clone could not use the one
# correction measured to flip gold separation from negative to positive. Precomputing it into the
# pool costs one float per candidate and removes the dependency entirely.
BC_SRC = os.environ.get("BC_SRC", "overlap")
# DELTA charges a candidate for repeating one already selected. 0 = plain top-k (unchanged).
DELTA     = float(os.environ.get("DELTA", "0"))
# NOVELTY charges a candidate for repeating the question's own wording.
NOV       = float(os.environ.get("NOVELTY", "0"))
# SYN charges a candidate for being a rewording of the seed it was reached from.
#
# NOVELTY compares against the question text and misses the cases that matter: the vignette says
# "somnolent", the candidate is called "Excessive somnolence", and no token matches, so the slot
# survives while genuinely relevant slots that happen to share a word get dropped. Comparing a
# candidate to its OWN origin_seed has no such gap — morphology is irrelevant when both strings
# come from the same concept.
#
# The current ranking is strongly biased toward these: 26.9% of pool candidates have
# syn >= 0.5 against 62.1% of the slots that reach the prompt, a 2.3x enrichment. Walking two hops
# to arrive at a synonym of the starting point tells the reader nothing it did not already have.
SYN       = float(os.environ.get("SYN", "0"))
# SRC mixes two retrieval sources inside one ranking instead of merging two blocks.
#
# On the union pool the walker's candidates and raw_1hop's are scored on the same scale, and the
# ranking still reproduces the walker's block almost exactly -- because cos measures similarity to
# the SYMPTOM phrases, and a disease name is structurally less similar to symptom text than a
# symptom synonym is. The information is in the pool; cos/bc/hop cannot see it. SRC is the one
# remaining term that carries something none of them do: which traversal found the candidate.
#
# 0 reproduces the walker ranking exactly; large values reproduce raw_1hop's ordering. It is a
# crude mixing parameter and is labelled as one -- its justification is that the two sources were
# read to fail in complementary ways, not that the number means anything on its own.
SRC       = float(os.environ.get("SRC", "0"))
# COS_NORM=source z-scores cos WITHIN each retrieval source before it is used.
#
# The two sources put their candidates on different scales: on the union pool the walker's
# candidates have median cos 0.47 (329) / 0.52 (medbullets), raw_1hop's have 0.32 / 0.33, and 80%
# of raw_1hop's fall below the walk's own tau of 0.40. Entry to the top-10 needs about 0.62. So
# raw_1hop's candidates cannot compete on raw cos no matter what the coefficients are -- and they
# are the ones carrying the gold: 4.9% of them are gold against 1.55% of the walker's, a 3.2x
# higher density at 0.15 lower cos.
#
# Comparing raw cos across the two is the same error as comparing sigma_bc to sigma_cos without
# normalising: the number that wins is decided by the scale, not by the information. Z-scoring per
# source asks instead "how good is this candidate FOR ITS SOURCE", which is the comparison the two
# distributions actually support.
COS_NORM  = os.environ.get("COS_NORM", "")
# ABSTRACT charges a candidate for being a navigational node rather than a clinical concept.
#
# Reading 25 blocks side by side, raw_1hop's characteristic failure was not a wrong disease but no
# disease at all: `Inflammation`, `Infectious process`, `Anatomical or acquired body structure`,
# `Organ part`, `Damage`, `Increased`. Measured, those are 15.2% of its slots plus 5.7% single-word
# generics, against 10.4% and 0.5% for the walker. Letting raw_1hop's candidates compete (COS_NORM
# or SRC) imports that failure mode along with its coverage -- abstract slots go 0.11 -> 0.38 per
# block on MedBullets -- so the two belong together.
#
# The test is deliberately narrow: a UMLS semantic-type marker AND at most two content words. That
# keeps `Acute angle closure glaucoma (disorder)` while dropping `Anatomical structure (body
# structure)`.
# SRC_QUOTA reserves slots for the second retrieval source, ranked by ITS OWN order.
#
# Every attempt to let the two sources compete on one score failed for the same reason: their cos
# distributions do not overlap usefully (walker median 0.47/0.52 vs raw_1hop 0.32/0.33, entry to
# the top-10 needs ~0.62), and z-scoring them onto a common scale imports raw_1hop's navigational
# nodes along with its coverage. The comparison the data does NOT support is exactly the one those
# designs require.
#
# A quota asks nothing of the sort. It takes the best k FOR ITS SOURCE and the rest from the other,
# which is what the shipped role quota already does for disease vs non-disease -- and that quota was
# measured to help (removing it inflated finding slots 29% -> 45% and cost 1.5-2.3pp of gold@10).
SRC_QUOTA = int(os.environ.get("SRC_QUOTA", "0"))
# DDX_QUOTA reserves slots for candidates reached from a DIAGNOSIS seed.
#
# The source quota asked "which script found this", which is a property of the plumbing. What the
# content actually distinguished was where a candidate came FROM: raw_1hop's good blocks were the
# neighbourhood of the LLM's differential (all CLL variants, all SVT variants) while the walker's
# bad ones were the neighbourhood of an incidental descriptor (somnolence, muscular fatigue). The
# seed's own role says which is which, and 62-64% of seeds resolve to a disease, so there is enough
# material for a quota to draw on.
DDX_QUOTA = int(os.environ.get("DDX_QUOTA", "0"))
ABSTRACT  = float(os.environ.get("ABSTRACT", "0"))
_ABS_RE = re.compile(r"\b(structure|morphologic abnormality|qualifier value|situation|"
                     r"context-dependent|observable entity|event|process|system|region|part|"
                     r"inflammation|damage|increased|decreased|abnormal|navigational|"
                     r"attribute|organism|agent)\b", re.I)


def is_abstract(c):
    n = c.get("name") or ""
    return bool(_ABS_RE.search(n)) and len(name_tokens(n)) <= 2


def syn_of(c):
    """Lexical overlap between a candidate's name and the seed it was reached from."""
    a, b = name_tokens(c.get("name")), name_tokens(c.get("origin_seed"))
    return _overlap(a, b)

W_MAX  = float(os.environ.get("W_MAX", "0.6"))
# WEIGHT_FORM=additive drops the convex constraint on criticality / dispersion / adaptive.
#
# The convex form (1-w)*cos + w*bc forces two claims to move together: "time matters more here"
# can only be expressed as "and semantics matter less". Nothing about either mechanism's rationale
# requires that trade -- a question can need both. Worse, shrinking cos to (1-w) while mu*hop keeps
# its full size silently scales the hop penalty by 1/(1-w), which is where the HOP_SCALED confound
# came from: a temporal-looking knob that was partly a hop knob.
#
# additive keeps cos at full weight and lets the judgement set lambda per question:
#     cos + LAMBDA_MAX * r(q) * bc - mu * hop,     r(q) = the mechanism's output rescaled to [0,1]
# LAMBDA_MAX defaults to 2*LAMBDA so that r = 0.5 reproduces the shipped fixed lambda exactly.
WEIGHT_FORM = os.environ.get("WEIGHT_FORM", "convex")
LAMBDA_MAX  = float(os.environ.get("LAMBDA_MAX", str(2 * float(os.environ.get("LAMBDA", "0.3")))))
# below this many bc>0 candidates a question has no temporal spread to measure
DISP_MIN_BC = int(os.environ.get("DISP_MIN_BC", "5"))
# DISP_HOP=1 puts hop into the same dispersion split as cos and bc, so all three weights come
# from one rule and μ is not used at all. Off by default only so the two-signal variants already
# measured stay reproducible.
DISP_HOP = os.environ.get("DISP_HOP", "") not in ("", "0")
H_MAX = 2                     # walker max_hops; hop ∈ {0,1,2}


def hop_term(c):
    """Closeness, not penalty: 1 at the seed, 0 at the deepest hop."""
    return 1.0 - c.get("hop", 0) / H_MAX
HOP_SCALED = os.environ.get("HOP_SCALED", "") not in ("", "0")
# JUDGE is an extension point, not a mode. MODE picks one of the weight rules written here; JUDGE
# hands the whole weighting decision to an external module, so a new rule can be tried without
# touching this file. `judged_utility.py` (a separate line of work, not in this repo) is one such
# module — the error below says so rather than pretending the option does not exist, because the
# frozen files it produced are named in the results and a reader will meet them.
JUDGE  = os.environ.get("JUDGE", "")
NORMALIZE = os.environ.get("NORMALIZE", "1") not in ("", "0")
MAG_NORM  = os.environ.get("MAG_NORM", "1") not in ("", "0")
CRIT   = {}
if MODE == "criticality":
    p = PIPE / f"datasets/{DS}/criticality.json"
    if not p.exists():
        raise SystemExit(f"沒有 {p} — 先跑 DATASET={DS} python3 pipeline/extract_duration_criticality.py")
    CRIT = {k: v["score"] for k, v in json.load(open(p)).items()}

# adaptive's breakpoints come from the pool's own cos distribution, not from a guessed constant.
# A hardcoded "cos > 0.8 means confident" would essentially never fire: cos>0.8 is 0.15–0.34% of
# candidates and only 15–22% of questions have even one.
_C_LO = _C_HI = None
def _cos_breakpoints(records):
    v = sorted(c["cos"] for r in records for c in r["candidates"])
    if not v:
        return 0.4, 0.63
    pick = lambda p: v[min(int(p * len(v)), len(v) - 1)]
    return pick(0.25), pick(0.90)


# MODE=dispersion asks each question's own candidate set which signal is worth listening to.
#
# A signal that is nearly constant across the candidates cannot separate them, whatever its
# average level — it contributes the same amount to every score and the ranking is decided by
# whatever else varies. So the weight follows the spread: the signal that actually moves across
# THIS question's neighbours gets to steer, per question rather than per candidate.
#
# Raw σ is not comparable between the two signals. bc lives in 0–1 with a non-zero median of 0.67;
# cos sits in a narrow 0.35–0.65 band. Comparing their standard deviations directly would hand bc
# the weight on nearly every question for a reason that has nothing to do with information. Each
# question's σ is therefore replaced by its PERCENTILE within that signal's own cross-question σ
# distribution, which makes "this question's cos is unusually spread out" comparable to the same
# statement about bc.
_DISP = {}


def _dispersion_weights(records):
    """w_bc per question, from the percentile-normalised spread of bc against that of cos.

    bc == 0 means three different things — no patient duration, a non-disease node, or genuine
    temporal incompatibility. Counting the zeros in σ_bc would read "unknown" as "incompatible" and
    report a spread that is really a missing-data pattern, so σ_bc is taken over the bc > 0
    candidates only. Below MIN_BC of them there is not enough temporal signal on this question to
    have an opinion, and w_bc is 0 — the honest version of the shipped formula, which multiplies
    an all-zero bc by 0.3 on 46% of MedBullets questions.
    """
    sig = {}
    for r in records:
        cs = [c.get("cos", 0.0) for c in r["candidates"]]
        bs = [b for b in (_bc(c, r["uid"]) for c in r["candidates"]) if b > 0]
        hs = [hop_term(c) for c in r["candidates"]]
        sig[r["uid"]] = (st.pstdev(cs) if len(cs) > 1 else 0.0,
                         st.pstdev(bs) if len(bs) >= DISP_MIN_BC else None,
                         st.pstdev(hs) if len(hs) > 1 else 0.0)
    cos_ref = sorted(v[0] for v in sig.values())
    bc_ref = sorted(v[1] for v in sig.values() if v[1] is not None)
    hop_ref = sorted(v[2] for v in sig.values())

    def pct(ref, x):
        if not ref:
            return 0.0
        lo = bisect.bisect_left(ref, x)
        return (lo + bisect.bisect_right(ref, x)) / (2 * len(ref))

    out = {}
    for uid, (sc, sb, sh) in sig.items():
        pc = pct(cos_ref, sc)
        pb = 0.0 if sb is None else pct(bc_ref, sb)
        ph = pct(hop_ref, sh)
        if not DISP_HOP:
            # two-signal split; hop stays on the shipped −μ·hop penalty
            out[uid] = W_MAX * (pb / (pc + pb)) if (pc + pb) > 0 else 0.0
            continue
        # three-signal split: μ disappears. hop enters as the CLOSENESS term (1 − hop/H) so that
        # all three point the same way and their coefficients are on one scale — a subtracted
        # penalty sitting beside two added rewards cannot be compared to them, and a rule that
        # cannot compare its inputs is not deciding anything.
        #
        # Scale reference, because a_hop is otherwise hard to read: within a question the constant
        # part of a_hop·(1 − hop/H) does not change the order, so a_hop is equivalent to a per-hop
        # penalty of a_hop/H. a_hop = 0.16 reproduces the shipped μ = 0.08.
        tot = pc + pb + ph
        out[uid] = ({"cos": pc / tot, "bc": pb / tot, "hop": ph / tot} if tot > 0
                    else {"cos": 1.0, "bc": 0.0, "hop": 0.0})
    return out


def weight_for(c, uid):
    if MODE == "criticality":
        return min(CRIT.get(uid, 0.0), W_MAX)
    if MODE == "dispersion":
        return _DISP.get(uid, 0.0)
    if MODE == "adaptive":
        # linear ramp: cos ≤ p25 → full temporal weight, cos ≥ p90 → none
        span = max(_C_HI - _C_LO, 1e-6)
        return W_MAX * min(max((_C_HI - c.get("cos", 0.0)) / span, 0.0), 1.0)
    return None


_CN = {}
# seed -> role, read off the hop-0 candidates: a hop-0 candidate IS the seed it came from
_SEED_ROLE = {}
for _r in records:
    for _c in _r["candidates"]:
        if _c.get("hop", 0) == 0:
            _SEED_ROLE[(_r["uid"], _c.get("origin_seed"))] = _c.get("role")


def _cos(c):
    """cos, optionally z-scored within the candidate's own retrieval source."""
    v = c.get("cos", 0.0)
    if COS_NORM != "source":
        return v
    m, sd = _CN.get(c.get("src", "walker"), (0.0, 1.0))
    return (v - m) / sd


def _bc(c, uid):
    if BC_SRC != "onesided":
        return c.get("bc", 0.0)
    if "bc_onesided" not in c:
        raise SystemExit("此池沒有 bc_onesided 欄位 — 需要重新以 build_pools.py 產生")
    return c["bc_onesided"]


def utility(c, uid=None):
    base = _utility_raw(c, uid)
    if SYN:
        base -= SYN * syn_of(c)
    if SRC and c.get("src") == "raw_1hop":
        base += SRC
    if ABSTRACT and is_abstract(c):
        base -= ABSTRACT
    if NOV:
        base += NOV * novelty(c, _QTOK.get(uid, set()))
    return base


def _utility_raw(c, uid=None):
    if MODE == "dispersion" and DISP_HOP:
        a = _DISP.get(uid) or {"cos": 1.0, "bc": 0.0, "hop": 0.0}
        return a["cos"] * c.get("cos", 0.0) + a["bc"] * _bc(c, uid) + a["hop"] * hop_term(c)
    if JUDGE:
        return _JUDGE_OBJ.utility(c, uid)
    if UTILITY:
        return eval(UTILITY, {"__builtins__": {}},
                    {"cos": c.get("cos", 0.0), "bc": c.get("bc", 0.0),
                     "hop": c.get("hop", 0), "score": c.get("score", 0.0)})
    bcv = _bc(c, uid)
    w = weight_for(c, uid)
    if w is None:
        return _cos(c) + LAMBDA * bcv - MU * c.get("hop", 0)
    # HOP_SCALED=1 keeps cos:hop fixed as w moves.
    #
    # Without it the convex form is confounded. It shrinks cos to (1−w) but leaves μ·hop at full
    # size, so raising w strengthens the hop penalty by 1/(1−w) as a side effect — and hop is the
    # ONE component with real gold/distractor separation (5.8% gold at hop 0 vs 0.22% at hop 2,
    # where cos and bc both separate the wrong way). An uncorrected criticality variant can
    # therefore win by acting as a hop-penalty knob while appearing to be a temporal one. The
    # tell: on questions with no duration, where bc is 0 for every candidate, the two still
    # ranked differently (only 5/24 and 64/143 blocks matched).
    if WEIGHT_FORM == "additive":
        r = min(max(w / W_MAX, 0.0), 1.0) if W_MAX else 0.0
        return _cos(c) + LAMBDA_MAX * r * bcv - MU * c.get("hop", 0)
    hop_w = (1 - w) if HOP_SCALED else 1.0
    return (1 - w) * _cos(c) + w * bcv - MU * hop_w * c.get("hop", 0)


def rank(cands, uid=None):
    """Score-sort, apply the role quota, then cut to TOP_K — the order format_kg_block uses."""
    if not cands:
        return []
    if SAMPLE == "random":
        # seeded through hashlib, not the built-in hash(): Python randomises string hashing per
        # process, so hash(uid) would draw a different sample on every run while claiming to be
        # reproducible — the same defect that made the interval_sample ablation unrepeatable
        import hashlib, random
        rng = random.Random(int.from_bytes(
            hashlib.sha256(f"{DS}|{METHOD}|{uid}".encode()).digest()[:8], "big"))
        return rng.sample(cands, min(TOP_K, len(cands)))
    # MAX_HOP is a hard cut, not a penalty. The gold-separation audit found gold concentrated at
    # hop 0–1 (40/43/20% for 329, 56/36/8% for medbullets) while distractors sit at hop 2 (74%),
    # so a hop-2 candidate is gold 0.22% of the time against 5.8% at hop 0. μ trades that off
    # against score; this removes the region outright, which is structurally what raw_1hop does —
    # and raw_1hop currently outranks the walker on both new datasets.
    if MAX_HOP is not None:
        kept = [c for c in cands if c.get("hop", 0) <= MAX_HOP]
        if kept:                      # never empty the block: a question with only deep hits
            cands = kept              # keeps its (weak) evidence rather than falling back to no-KG
    # DELTA used to return here, which silently skipped the DDX_QUOTA branch below: a variant named
    # `_d0.3_ab0.6_ddx7` was really "diversity + the OLD role quota", and its DDx-seeded slot count
    # was 5.3-5.9 against ddx7's 6.7-6.8. Diversity now runs INSIDE each quota group instead, so the
    # two compose rather than one overriding the other.
    if scored and DELTA > 0 and not (DDX_QUOTA or SRC_QUOTA):
        picked = select_diverse(cands, lambda c: utility(c, uid), TOP_K, DELTA, Q_DIS, Q_OTHER)
        if picked is not None:
            picked.sort(key=lambda c: -utility(c, uid))
            return picked[:TOP_K]
    if scored:
        cands = sorted(cands, key=lambda c: -utility(c, uid))
    if scored and DDX_QUOTA > 0:
        dq = [c for c in cands if _SEED_ROLE.get((uid, c.get("origin_seed"))) == "disease"]
        rest = [c for c in cands if _SEED_ROLE.get((uid, c.get("origin_seed"))) != "disease"]
        if ABSTRACT:
            k2 = [c for c in dq if not is_abstract(c)]
            dq = k2 or dq
        dq.sort(key=lambda c: -utility(c, uid)); rest.sort(key=lambda c: -utility(c, uid))
        k = min(DDX_QUOTA, TOP_K)
        if DELTA > 0:
            # greedy within each group, but the redundancy penalty sees BOTH groups' picks --
            # otherwise the two halves of the block can still restate each other
            picked = []
            for grp, want in ((dq, k), (rest, TOP_K - k)):
                avail = list(grp)
                while avail and sum(1 for c in picked if c in grp) < want:
                    best, bv = None, None
                    for c in avail:
                        pen = max((_overlap(name_tokens(c.get("name")), name_tokens(p.get("name")))
                                   for p in picked), default=0.0)
                        v = utility(c, uid) - DELTA * pen
                        if bv is None or v > bv:
                            best, bv = c, v
                    picked.append(best); avail.remove(best)
        else:
            picked = dq[:k] + rest[:TOP_K - k]
        seen = {c.get("cui") or c.get("name") for c in picked}
        for c in dq[k:] + rest[TOP_K - k:]:
            if len(picked) >= TOP_K: break
            if (c.get("cui") or c.get("name")) in seen: continue
            picked.append(c); seen.add(c.get("cui") or c.get("name"))
        picked.sort(key=lambda c: -utility(c, uid))
        return picked[:TOP_K]
    if scored and SRC_QUOTA > 0 and any(c.get("src") == "raw_1hop" for c in cands):
        # rank each source by its own utility, then interleave under the quota
        alt = [c for c in cands if c.get("src") == "raw_1hop"]
        own = [c for c in cands if c.get("src") != "raw_1hop"]
        if ABSTRACT:
            keep = [c for c in alt if not is_abstract(c)]
            alt = keep or alt
        alt.sort(key=lambda c: -utility(c, uid))
        own.sort(key=lambda c: -utility(c, uid))
        k = min(SRC_QUOTA, TOP_K)
        picked = own[:TOP_K - k] + alt[:k]
        seen = {c.get("cui") or c.get("name") for c in picked}
        for c in own[TOP_K - k:] + alt[k:]:
            if len(picked) >= TOP_K: break
            if (c.get("cui") or c.get("name")) in seen: continue
            picked.append(c); seen.add(c.get("cui") or c.get("name"))
        return picked[:TOP_K]
    if scored and (Q_DIS or Q_OTHER):
        dis = [c for c in cands if c.get("role") == "disease"]
        oth = [c for c in cands if c.get("role") != "disease"]
        picked = dis[:Q_DIS] + oth[:Q_OTHER]
        seen = {c.get("cui") or c.get("name") for c in picked}
        for c in cands:
            if len(picked) >= TOP_K: break
            if (c.get("cui") or c.get("name")) in seen: continue
            picked.append(c)
        picked.sort(key=lambda c: -utility(c, uid))
    else:
        picked = cands
    return picked[:TOP_K]


def block(cands, uid=None):
    if not cands:
        return "  (no paths found)"
    out = []
    for i, c in enumerate(cands, 1):
        if "chain" in c:
            out.append(f"  {i}. {c['chain']}")
        elif "cos" in c:
            # print the bc that ACTUALLY scored this candidate. Showing the stored overlap next
            # to a one-sided ranking would hand the reader a number that does not explain the order
            # it is looking at.
            out.append(f"  {i}. [{c['role']}] {c['name'][:60]} "
                       f"(score={utility(c, uid):.2f}: cos={c['cos']:.2f}+bc={_bc(c, uid):.2f})")
        else:
            out.append(f"  {i}. {c['name']}")
    return "\n".join(out)


def days_to_phrase(d):
    if isinstance(d, (list, tuple)):
        seen = []
        for x in d:
            p = days_to_phrase(x)
            if p not in seen: seen.append(p)
        return " / ".join(seen)
    if d < 1:   return f"{int(d*24)} hours"
    if d < 14:  return f"{int(d)} days"
    if d < 60:  return f"{int(d/7)} weeks"
    if d < 365: return f"{int(d/30)} months"
    return f"{int(d/365)} years"


_JUDGE_OBJ = None
if JUDGE:
    try:
        from judged_utility import Judge
    except ImportError:
        raise SystemExit(
            f"JUDGE={JUDGE} 需要 judged_utility.py，此 repo 未附。"
            " 若要自訂加權規則，實作一個提供 utility(candidate, uid) 的物件並在此接上即可。")
    _JUDGE_OBJ = Judge(JUDGE, records, DS, lambda c: _bc(c, None),
                       normalize=NORMALIZE, mag_norm=MAG_NORM)

if COS_NORM == "source":
    import collections as _co
    _by = _co.defaultdict(list)
    for _r in records:
        for _c in _r["candidates"]:
            _by[_c.get("src", "walker")].append(_c.get("cos", 0.0))
    for _k, _v in _by.items():
        _m = st.fmean(_v); _s = st.pstdev(_v) or 1.0
        _CN[_k] = (_m, _s)
        print(f"cos 來源內標準化: {_k:10s} n={len(_v):6d} 中位前 μ={_m:.3f} σ={_s:.3f}")

if MODE == "dispersion":
    _DISP = _dispersion_weights(records)
    if DISP_HOP:
        _q = lambda k: sorted(v[k] for v in _DISP.values())
        _m = lambda k: _q(k)[len(_DISP)//2]
        print(f"dispersion 三訊號權重（逐題，μ 未使用）中位 a_cos={_m('cos'):.3f} "
              f"a_bc={_m('bc'):.3f} a_hop={_m('hop'):.3f}  "
              f"| a_hop 等效每 hop 懲罰 {_m('hop')/H_MAX:.3f}（shipped μ=0.08）")
    else:
        _w = sorted(_DISP.values())
        _nz = [x for x in _w if x > 0]
        print(f"dispersion 逐題 w_bc: {len(_nz)}/{len(_w)} 題有時間訊號, "
              f"中位 {(_w[len(_w)//2]):.3f}, 有訊號者中位 {(_nz[len(_nz)//2] if _nz else 0):.3f}, "
              f"max {(_w[-1] if _w else 0):.3f}")

if MODE == "adaptive":
    _C_LO, _C_HI = _cos_breakpoints(records)
    print(f"adaptive 斷點取自本池的 cos 分布: p25={_C_LO:.3f} → w={W_MAX:.2f}, "
          f"p90={_C_HI:.3f} → w=0")

bench = {x["uid"]: x for x in json.load(open(PIPE / f"datasets/{DS}/benchmark.json"))}
_QTOK = {u: question_tokens(x["question"]) for u, x in bench.items()} if NOV else {}
tpl = open(PIPE.parent / TEMPLATE_FILE if TEMPLATE_FILE.startswith("pipeline/")
           else TEMPLATE_FILE).read() if TEMPLATE_FILE else None

items, sizes = [], []
for r in records:
    it = bench.get(r["uid"])
    if not it: continue
    picked = rank(r["candidates"], r["uid"])
    sizes.append(len(picked))
    kg = block(picked, r["uid"])
    ob = "\n".join(f"  {k}. {v}" for k, v in sorted(it["options"].items()))
    pd = r.get("patient_days")
    dur = days_to_phrase(pd) if pd else P.NO_DURATION_STR
    if tpl:
        pr = tpl.format(question=it["question"], options=ob,
                        patient_duration=dur, retrieved_information=kg)
    elif picked:
        pr = P.WALKER.format(question=it["question"], options_block=ob,
                             patient_dur_str=dur, kg_block=kg)
    else:
        pr = P.NO_KG.format(question=it["question"], options_block=ob)
    items.append({"uid": r["uid"], "gold": r["gold"], "route": r.get("route"),
                  "kg_block": kg if picked else "", "prompt": pr})

if MODE == "fixed":
    _wtag = f"_l{LAMBDA:g}_m{MU:g}"
else:
    _wtag = f"_{MODE}_w{W_MAX:g}_m{MU:g}"
if JUDGE:
    _wtag = "_j" + JUDGE + ("" if NORMALIZE else "_nonorm") + ("" if MAG_NORM else "_magnone")
variant = TAG or (f"k{TOP_K}" + ("" if not scored else _wtag)
                  + (f"_h{MAX_HOP}" if MAX_HOP is not None else "")
                  + ("_hs" if HOP_SCALED and MODE != "fixed" else "")
                  + ("_add" if WEIGHT_FORM == "additive" and MODE != "fixed" else "")
                  + ("_h3" if MODE == "dispersion" and DISP_HOP else "")
                  + ("_rand" if SAMPLE == "random" else "")
                  + (f"_d{DELTA:g}" if DELTA > 0 else "")
                  + (f"_n{NOV:g}" if NOV else "")
                  + (f"_s{SYN:g}" if SYN else "")
                  + (f"_src{SRC:g}" if SRC else "")
                  + ("_zsrc" if COS_NORM == "source" else "")
                  + (f"_ab{ABSTRACT:g}" if ABSTRACT else "")
                  + (f"_q{SRC_QUOTA}" if SRC_QUOTA else "")
                  + (f"_ddx{DDX_QUOTA}" if DDX_QUOTA else "")
                  + ("_os" if BC_SRC == "onesided" else "")
                  + ("_u" + UTILITY.replace(" ", "") if UTILITY else ""))
name = f"{METHOD}__{variant}"
med = sorted(sizes)[len(sizes)//2] if sizes else 0
print(f"{DS}/{METHOD}: {len(items)} 題，每題證據中位數 {med} 條"
      f"（TOP_K={TOP_K}"
      + ("" if not scored else f", λ={LAMBDA}, μ={MU}")
      + (f", utility={UTILITY}" if UTILITY else "") + ")")
print(f"  變體名稱: {name}")
if DRY:
    print("\n  DRY=1，未寫檔。範例 kg_block:\n")
    print("\n".join("    " + l for l in items[0]["kg_block"].splitlines()))
else:
    out = PIPE / f"frozen/{DS}/{name}.json"
    json.dump({"method": name, "dataset": DS, "n": len(items),
               "from_pool": str(pool_path.relative_to(PIPE)),
               "params": {"top_k": TOP_K, "mode": MODE, "w_max": W_MAX, "weight_form": WEIGHT_FORM, "lambda_max": LAMBDA_MAX, "max_hop": MAX_HOP, "hop_scaled": HOP_SCALED, "bc_src": BC_SRC,
                          "judge": JUDGE or None, "normalize": NORMALIZE, "mag_norm": MAG_NORM,
                          "delta": DELTA, "novelty": NOV, "syn": SYN, "src_bonus": SRC, "cos_norm": COS_NORM or None, "abstract": ABSTRACT, "src_quota": SRC_QUOTA, "ddx_quota": DDX_QUOTA,
                          "cos_p25": _C_LO, "cos_p90": _C_HI,
                          "lambda_bc": LAMBDA, "mu_hop": MU,
                          "utility": UTILITY or "cos + λ·bc − μ·hop",
                          "quota_disease": Q_DIS, "quota_other": Q_OTHER},
               "items": items}, open(out, "w"), indent=1)
    print(f"  -> frozen/{DS}/{name}.json")
    print(f"  下一步: DATASET={DS} METHOD={name} N_RUNS=2 RESULTS_DIR=results/param_sweep "
          f"python3 pipeline/run_reader.py")
