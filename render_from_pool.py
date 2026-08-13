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
import json, os, importlib.util
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
BC_SRC = os.environ.get("BC_SRC", "overlap")
if BC_SRC == "onesided":
    import sys as _sys; _sys.path.insert(0, str(PIPE))
    from bc_onesided import compat_for_cui as _compat
W_MAX  = float(os.environ.get("W_MAX", "0.6"))
HOP_SCALED = os.environ.get("HOP_SCALED", "") not in ("", "0")
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


def weight_for(c, uid):
    if MODE == "criticality":
        return min(CRIT.get(uid, 0.0), W_MAX)
    if MODE == "adaptive":
        # linear ramp: cos ≤ p25 → full temporal weight, cos ≥ p90 → none
        span = max(_C_HI - _C_LO, 1e-6)
        return W_MAX * min(max((_C_HI - c.get("cos", 0.0)) / span, 0.0), 1.0)
    return None


def _bc(c, uid):
    if BC_SRC != "onesided":
        return c.get("bc", 0.0)
    t = _PD.get(uid)
    # non-disease roles keep bc=0: temporal compatibility is undefined for a drug or a procedure,
    # and 0 there means "not applicable", exactly as in the shipped scorer
    return _compat(c["cui"], t) if (t and c.get("role") == "disease") else 0.0


def utility(c, uid=None):
    if UTILITY:
        return eval(UTILITY, {"__builtins__": {}},
                    {"cos": c.get("cos", 0.0), "bc": c.get("bc", 0.0),
                     "hop": c.get("hop", 0), "score": c.get("score", 0.0)})
    bcv = _bc(c, uid)
    w = weight_for(c, uid)
    if w is None:
        return c.get("cos", 0.0) + LAMBDA * bcv - MU * c.get("hop", 0)
    # HOP_SCALED=1 keeps cos:hop fixed as w moves.
    #
    # Without it the convex form is confounded. It shrinks cos to (1−w) but leaves μ·hop at full
    # size, so raising w strengthens the hop penalty by 1/(1−w) as a side effect — and hop is the
    # ONE component with real gold/distractor separation (5.8% gold at hop 0 vs 0.22% at hop 2,
    # where cos and bc both separate the wrong way). An uncorrected criticality variant can
    # therefore win by acting as a hop-penalty knob while appearing to be a temporal one. The
    # tell: on questions with no duration, where bc is 0 for every candidate, the two still
    # ranked differently (only 5/24 and 64/143 blocks matched).
    hop_w = (1 - w) if HOP_SCALED else 1.0
    return (1 - w) * c.get("cos", 0.0) + w * bcv - MU * hop_w * c.get("hop", 0)


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
    if scored:
        cands = sorted(cands, key=lambda c: -utility(c, uid))
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


if MODE == "adaptive":
    _C_LO, _C_HI = _cos_breakpoints(records)
    print(f"adaptive 斷點取自本池的 cos 分布: p25={_C_LO:.3f} → w={W_MAX:.2f}, "
          f"p90={_C_HI:.3f} → w=0")

bench = {x["uid"]: x for x in json.load(open(PIPE / f"datasets/{DS}/benchmark.json"))}
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
variant = TAG or (f"k{TOP_K}" + ("" if not scored else _wtag)
                  + (f"_h{MAX_HOP}" if MAX_HOP is not None else "")
                  + ("_hs" if HOP_SCALED and MODE != "fixed" else "")
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
               "params": {"top_k": TOP_K, "mode": MODE, "w_max": W_MAX, "max_hop": MAX_HOP, "hop_scaled": HOP_SCALED, "bc_src": BC_SRC,
                          "cos_p25": _C_LO, "cos_p90": _C_HI,
                          "lambda_bc": LAMBDA, "mu_hop": MU,
                          "utility": UTILITY or "cos + λ·bc − μ·hop",
                          "quota_disease": Q_DIS, "quota_other": Q_OTHER},
               "items": items}, open(out, "w"), indent=1)
    print(f"  -> frozen/{DS}/{name}.json")
    print(f"  下一步: DATASET={DS} METHOD={name} N_RUNS=2 RESULTS_DIR=results/param_sweep "
          f"python3 pipeline/run_reader.py")
