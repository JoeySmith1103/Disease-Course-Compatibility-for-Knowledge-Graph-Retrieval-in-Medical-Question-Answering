#!/usr/bin/env python3
"""A judgement layer over the walker's ranking signals.

The shipped utility is `cos + 0.3·bc − 0.08·hop`. Its three constants encode an assumption that
does not hold: that semantic match and temporal compatibility are worth the same amount on every
question and every candidate. Measured, they are not — the LLM criticality judge puts the median
question at 0.50 on 329 but 0.20 on MedBullets and 0.10 on MMLU, and 46% of MedBullets questions
have no usable patient duration at all, so their bc is identically 0 and still gets multiplied by
0.3.

So instead of searching for better constants, each signal's weight is decided by a judge:

    U(c) = a_cos·cos(c) + a_bc·bc(c) + a_hop·(1 − hop(c)/H)

Three judges, usable alone or combined with `+`:

  llm         External. An LLM reads the vignette (never the options) and scores how much the
              diagnosis turns on the time course. Per question.
  dispersion  Distributional self-judgement. A signal that is tightly concentrated across this
              question's candidates cannot separate them, so it is worth less here. Per question.
  magnitude   Local self-judgement. For this candidate, which signal is actually speaking. Per
              candidate.

The hop term is rewritten as `1 − hop/H` so all three signals point the same way and share a
range. In the shipped form hop is a subtracted penalty sitting next to two added rewards, which
makes its coefficient incomparable to the other two — and a judge that cannot compare its inputs
is not judging anything.

Weights are NOT constrained to sum to 1. `NORMALIZE=1` divides by their sum; both variants are
built and the choice is made on measured accuracy, not on which looks tidier.
"""
import json, math, statistics as st
from pathlib import Path

PIPE = Path(__file__).resolve().parent
H_MAX = 2                    # walker's max_hops; hop ∈ {0,1,2}

# The shipped utility charges 0.08 per hop against a unit of cos. Here hop enters as the closeness
# term (1 − hop/H), so a weight of 0.08·H reproduces the same per-step cost — which is what the
# abstention path needs in order to rank a duration-free question exactly as the baseline does.
_SHIPPED_HOP_EQUIV = 0.08 * H_MAX


# ── redundancy ───────────────────────────────────────────────────────────────

_STOP = {"of", "the", "a", "an", "and", "or", "to", "in", "with", "disorder", "finding",
         "disease", "syndrome", "nos", "due", "by", "on", "structure", "product", "substance"}
_WORD = __import__("re").compile(r"[a-z0-9]+")


def name_tokens(name):
    return {w for w in _WORD.findall((name or "").lower()) if w not in _STOP and len(w) > 2}


def _overlap(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b)
    return 0.0 if not i else 2 * i / (len(a) + len(b))


def select_diverse(cands, score, top_k, delta, q_dis=7, q_other=3):
    """Greedy selection that charges a candidate for repeating one already chosen.

    Measured on the shipped top-10: 4.6 slots per question on 329 and 5.3 on MedBullets overlap
    another slot by at least 0.5 on name tokens. Half the block restates the presenting complaint —
    "Excessive somnolence", "Somnolence syndrome", "Disorders of excessive somnolence", "Daytime
    somnolence" — and those four slots carry the information of one.

    No weighting can fix that: near-duplicates have nearly identical cos, so they rise and fall
    together whatever the coefficients are. The redundancy has to be charged at SELECTION time,
    which is what this does — the standard maximal-marginal-relevance trade-off, with name-token
    overlap standing in for a pairwise similarity we do not have stored.

    delta=0 reproduces the plain top-k ordering exactly, so the mechanism is a strict extension.
    """
    if delta <= 0:
        return None                      # caller falls back to the existing path
    pool = sorted(cands, key=lambda c: -score(c))
    toks = {id(c): name_tokens(c.get("name")) for c in pool}
    picked, n_dis = [], 0
    while pool and len(picked) < top_k:
        best, best_v = None, None
        for c in pool:
            # the role quota is preserved: it is what keeps symptom-like findings from taking the
            # whole block, and it solves a different problem from redundancy
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


# ── novelty ──────────────────────────────────────────────────────────────────

_Q_STOP = _STOP | {"patient", "history", "year", "old", "man", "woman", "presents", "year-old"}


def question_tokens(question):
    return {w for w in _WORD.findall((question or "").lower())
            if w not in _Q_STOP and len(w) > 2}


def novelty(c, q_toks):
    """1 − (share of this candidate's name already present in the question).

    A concept that restates the complaint tells the reader nothing it does not already have.
    Measured on the pools, the gold rate falls monotonically as that overlap rises:

        overlap 0        1.12% gold (329)   0.50% (medbullets)
        overlap 0–0.33   0.50%              0.22%
        overlap 0.34–.66 0.34%              0.14%
        overlap >0.66    0.27%              0.18%

    A 4.1× / 2.8× spread across four ordered bands — cleaner than anything else available here:
    cos separates gold from distractors by −0.022 / −0.087 (the wrong way), and bc likewise.

    This is not redundant with cos, it is opposed to it. cos rewards resembling the query, and the
    concepts that resemble the query most are the ones restating it — which is why the shipped
    top-10 fills with rewordings of the presenting complaint ("Chest wall pain", "Central chest
    pain", "Tenderness of chest"). Charging for that overlap corrects a bias cos creates.
    """
    return 1.0 - len(name_tokens(c.get("name")) & q_toks) / max(len(name_tokens(c.get("name"))), 1)


def hop_term(c):
    """Closeness, not distance: 1 at the seed, 0 at the horizon."""
    return 1.0 - min(c.get("hop", 0), H_MAX) / H_MAX


# ── dispersion judge ─────────────────────────────────────────────────────────

def _pstdev(v):
    return st.pstdev(v) if len(v) >= 2 else 0.0


def _sigmas_for_question(cands, bc_of):
    """Within-question spread of each signal.

    bc == 0 is excluded from the bc spread. That value is overloaded — it means "no duration
    stored for this CUI", "not a disease role", and "temporally incompatible" all at once (14.7%
    of candidates on 329, 38.1% on MedBullets, 25.8% on MMLU). Counting unknowns as zeros would
    manufacture spread where there is no measurement.
    """
    cos = [c["cos"] for c in cands]
    bc = [b for b in (bc_of(c) for c in cands) if b > 0]
    hop = [hop_term(c) for c in cands]
    return {"cos": _pstdev(cos),
            # fewer than 5 measured candidates is not a distribution; the judge abstains and the
            # weight falls out to 0 rather than resting on two points
            "bc": _pstdev(bc) if len(bc) >= 5 else 0.0,
            "hop": _pstdev(hop)}


def dispersion_weights(records, bc_of):
    """{uid: {signal: weight}} — each question's spread as a percentile of that signal's own
    across-question spread.

    Raw standard deviations are not comparable across these signals: bc spans 0–1 with a median
    around 0.67, cos is squeezed into roughly 0.35–0.65, and the hop term takes three values. A
    percentile against the signal's own distribution asks the only question that transfers — "is
    this signal unusually flat HERE?" — and is unit-free by construction.
    """
    per_q = {r["uid"]: _sigmas_for_question(r["candidates"], bc_of)
             for r in records if r.get("candidates")}
    ref = {s: sorted(v[s] for v in per_q.values()) for s in ("cos", "bc", "hop")}

    def pct(s, x):
        arr = ref[s]
        if not arr or x <= 0:
            return 0.0
        lo, hi = 0, len(arr)
        while lo < hi:                       # bisect_left without the import
            mid = (lo + hi) // 2
            if arr[mid] < x: lo = mid + 1
            else: hi = mid
        return lo / len(arr)

    return {uid: {s: pct(s, sig[s]) for s in ("cos", "bc", "hop")}
            for uid, sig in per_q.items()}


# ── magnitude judge ──────────────────────────────────────────────────────────

def magnitude_medians(records, bc_of):
    """Dataset-level medians used to put cos and bc on comparable footing."""
    cos = sorted(max(c["cos"], 0.0) for r in records for c in r.get("candidates", []))
    bc = sorted(b for r in records for c in r.get("candidates", []) if (b := bc_of(c)) > 0)
    m = lambda v: (v[len(v) // 2] if v else 1.0) or 1.0
    return {"cos": m(cos), "bc": m(bc)}


def magnitude_weights(c, bc_of, med, norm=True):
    """Per-candidate: how loudly is each signal speaking for THIS candidate?

    cos is clipped at 0 — it reaches −0.182 in the pools, and a negative weight would invert the
    signal rather than discount it. Without `norm` the comparison is biased before it starts: bc's
    median is 0.67 against cos's 0.49, so bc wins on scale alone. Dividing each by its own median
    makes "louder than usual for this signal" the thing being compared.
    """
    cp = max(c["cos"], 0.0)
    b = bc_of(c)
    if norm:
        cp, b = cp / med["cos"], b / med["bc"]
    return {"cos": cp, "bc": b, "hop": 1.0}


# ── llm judge ────────────────────────────────────────────────────────────────

def llm_scores(ds):
    p = PIPE / f"datasets/{ds}/criticality.json"
    if not p.exists():
        raise SystemExit(f"沒有 {p} — 先跑 DATASET={ds} python3 pipeline/extract_duration_criticality.py")
    return {k: float(v["score"]) for k, v in json.load(open(p)).items()}


# ── answer-type judge ────────────────────────────────────────────────────────

def answer_types(ds):
    p = PIPE / f"datasets/{ds}/answer_type.json"
    if not p.exists():
        raise SystemExit(f"沒有 {p} — 先跑 DATASET={ds} python3 pipeline/extract_answer_type.py")
    return {k: v["answer_type"] for k, v in json.load(open(p)).items()}


def answer_type_weight(c, uid, atypes, beta):
    """Up-weight candidates whose role matches what the question asks for.

    cos scores how much a concept reads like the vignette, and the vignette is a description of
    symptoms — so organisms, drugs and procedures are ranked by a signal that structurally
    disfavours them. On 329, 639 organism candidates in the pool yield 2 top-10 slots; on
    MedBullets, 211 yield none while 27% of questions ask for a procedure.

    beta is a free parameter, not the retired walker's +0.07. That constant was never tuned and
    applied only to organism and drug; here the target role comes from the question and the size of
    the correction is measured.
    """
    want = atypes.get(uid)
    if not want or want == "other":
        return 1.0
    role = c.get("role")
    # `finding` and `disease` are asked for far more often than they are missing from the top-10,
    # so boosting them mostly reshuffles candidates that were already there. The correction is
    # aimed at the roles cos suppresses.
    return (1.0 + beta) if role == want else 1.0


def llm_weights(uid, crit):
    """The external judge moves cos and bc against each other and leaves hop alone.

    hop is untouched deliberately: "how much does this question depend on time" says nothing about
    how far from the seeds a candidate should be trusted, and folding it in would let a temporal
    judgement silently retune graph distance — the confound that made the earlier criticality
    variant look effective for the wrong reason.
    """
    s = crit.get(uid, 0.0)
    return {"cos": 1.0 - s, "bc": s, "hop": 1.0}


# ── composition ──────────────────────────────────────────────────────────────

class Judge:
    """Combines the requested judges multiplicatively, then optionally normalises."""

    def __init__(self, spec, records, ds, bc_of, normalize=True, mag_norm=True,
                 abstain="shipped", beta=0.5):
        self.parts = [p for p in (spec or "").split("+") if p]
        self.normalize = normalize
        self.mag_norm = mag_norm
        self.bc_of = bc_of
        # abstain="shipped": a candidate with no temporal signal is ranked by the shipped
        #   cos − 0.08·hop, which guarantees the mechanism cannot lose where it knows nothing.
        # abstain="judged": the judge still arbitrates, but between cos and hop only. 0.08 was
        #   never tuned, and on MedBullets 45% of questions take this path, so pinning nearly half
        #   the corpus to an unexamined constant is a conservative choice, not an optimal one.
        self.abstain = abstain
        self.beta = beta
        self.disp = dispersion_weights(records, bc_of) if "dispersion" in self.parts else None
        self.crit = llm_scores(ds) if "llm" in self.parts else None
        self.med = magnitude_medians(records, bc_of) if "magnitude" in self.parts else None
        self.atype = answer_types(ds) if "atype" in self.parts else None

    def weights(self, c, uid):
        # Two stages, and the order matters.
        #
        # Stage 1 — how much is each signal worth here at all. dispersion and magnitude both
        # answer that question independently per signal, so they simply multiply.
        base = {"cos": 1.0, "bc": 1.0, "hop": 1.0}
        if self.disp is not None:
            d = self.disp.get(uid, {"cos": 0.0, "bc": 0.0, "hop": 0.0})
            for s in base: base[s] *= d[s]
        if self.med is not None:
            m = magnitude_weights(c, self.bc_of, self.med, self.mag_norm)
            for s in base: base[s] *= m[s]
        if self.atype is not None:
            base["cos"] *= answer_type_weight(c, uid, self.atype, self.beta)

        # Stage 2 — the external judge is a TRADE-OFF between semantics and time, so it may only
        # act where both sides exist. Applying its (1−s) factor to cos unconditionally means an
        # opinion about time still sets the cos:hop ratio on a question that carries no duration
        # at all — the same confound HOP_SCALED was patching. The pre-flight invariance check
        # caught it: with the factor applied unconditionally, M4 passed 20/23 silent questions and
        # M5 only 7/23.
        #
        # Where bc is unmeasurable, its share returns to cos rather than leaking into hop, and it
        # returns WITHOUT the judge's factor, so the result no longer depends on s.
        w = dict(base)
        if self.bc_of(c) > 0:
            if self.crit is not None:
                s = self.crit.get(uid, 0.0)
                w["cos"] = base["cos"] * (1.0 - s)
                w["bc"] = base["bc"] * s
            if self.normalize:
                tot = sum(w.values())
                if tot > 0:
                    w = {k: v / tot for k, v in w.items()}
            return w

        # No temporal evidence for this candidate — fall back to the SHIPPED ranking, exactly.
        #
        # "Return bc's share to cos" was not enough. It leaves a_hop untouched, so after
        # normalisation hop's share rises and the mechanism silently reranks on graph distance:
        # measured per-hop cost went from 0.08 to 0.25 (llm), 0.44 (magnitude) and 1.10
        # (dispersion) — up to 14× the shipped penalty. On MedBullets that hits 133 of 298
        # questions, which is where the llm mechanism lost 1.52pp.
        #
        # A judge with no evidence should abstain, not improvise a new cos:hop balance. Returning
        # the shipped coefficients makes the silent half identical to the baseline by
        # construction, so a mechanism can only win or lose where it actually has something to
        # judge.
        if self.abstain == "shipped":
            return {"cos": 1.0, "bc": 0.0, "hop": _SHIPPED_HOP_EQUIV}
        w = {"cos": base["cos"] + base["bc"], "bc": 0.0, "hop": base["hop"]}
        if self.normalize:
            tot = w["cos"] + w["hop"]
            if tot > 0:
                w = {"cos": w["cos"] / tot, "bc": 0.0, "hop": w["hop"] / tot}
        return w

    def utility(self, c, uid):
        # Abstention returns the shipped expression itself, not an equivalent reweighting of it.
        # Equal ranking was not enough: the closeness form adds a constant (0.16·(1−hop/H) vs
        # −0.08·hop), so the score printed next to each candidate differed from the baseline's on
        # questions the judge had explicitly declined to touch. The prompt is what the reader sees,
        # so "identical where the judge abstains" has to hold byte for byte, not just in order.
        if self.bc_of(c) <= 0 and self.abstain == "shipped":
            return c["cos"] - 0.08 * c.get("hop", 0)
        w = self.weights(c, uid)
        return w["cos"] * c["cos"] + w["bc"] * self.bc_of(c) + w["hop"] * hop_term(c)

    def describe(self, uid):
        """One line for the prompt, so the reader can interpret the score column it is shown."""
        names = {"llm": "how much this case hinges on time course (judged from the vignette)",
                 "dispersion": "how much each signal actually varies across the retrieved candidates",
                 "magnitude": "which signal is stronger for each individual candidate",
                 "atype": "what kind of answer the question asks for"}
        if not self.parts:
            return None
        why = "; ".join(names[p] for p in self.parts if p in names)
        return why
