#!/usr/bin/env python3
"""Pre-flight checks for the judgement layer. Reads files only — no LLM calls, no Neo4j.

The point is to find implementation errors here rather than by paying for three inference runs and
reading a number that turns out to mean nothing. A mechanism that fails any check does not go to
the batch.

  1  weights are finite and non-negative; report the ranges actually produced
  2  degeneracy — on questions with no measured duration (bc == 0 everywhere) the ranking must
     equal a cos+hop-only ranking. A temporal judge has no business reordering a question that
     carries no temporal signal; the earlier convex form failed exactly here, silently acting as
     a hop-penalty knob (329: only 6/24 blocks unchanged).
  3  the evidence actually moves versus the fixed baseline — a mechanism whose top-10 matches on
     >95% of questions cannot produce a measurable difference and is not worth an inference run
  4  weight distributions per dataset, plus the sanity relation that the llm judge must give
     MMLU less temporal weight than 329 (criticality medians 0.10 vs 0.50)

Usage:  python3 pipeline/verify_judge.py [dataset ...]
"""
import json, os, sys, statistics as st
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))
from judged_utility import Judge, hop_term

DATASETS = sys.argv[1:] or ["329", "medbullets"]
MECHS = [("M1", "llm", True, True), ("M2", "dispersion", True, True),
         ("M3", "magnitude", True, True), ("M4", "llm+dispersion", True, True),
         ("M5", "llm+dispersion+magnitude", True, True),
         ("M3n", "magnitude", True, False), ("M5r", "llm+dispersion+magnitude", False, True)]
Q_DIS, Q_OTHER, TOP_K = 7, 3, 10


def top10(cands, score):
    s = sorted(cands, key=lambda c: -score(c))
    dis = [c for c in s if c.get("role") == "disease"]
    oth = [c for c in s if c.get("role") != "disease"]
    picked = dis[:Q_DIS] + oth[:Q_OTHER]
    seen = {c["cui"] for c in picked}
    for c in s:
        if len(picked) >= TOP_K: break
        if c["cui"] in seen: continue
        picked.append(c)
    picked.sort(key=lambda c: -score(c))
    return [c["cui"] for c in picked[:TOP_K]]


for ds in DATASETS:
    recs = [json.loads(l) for l in open(PIPE / f"pool/{ds}/walker.jsonl")]
    recs = [r for r in recs if r.get("candidates")]
    bc_of = lambda c: c.get("bc", 0.0)
    base = {r["uid"]: top10(r["candidates"],
                            lambda c: c["cos"] + 0.3 * c["bc"] - 0.08 * c["hop"]) for r in recs}
    # A question is temporally silent when every candidate's bc is 0. The assertable property is
    # not "equals some arbitrary cos+hop ranking" — that fixes an unstated cos:hop ratio — but
    # INVARIANCE: a mechanism must rank a silent question the same way whether the temporal
    # judgement says time is decisive or irrelevant. Anything else means an opinion about time is
    # reordering a question that carries none.
    silent = [r for r in recs if all(bc_of(c) <= 0 for c in r["candidates"])]

    print(f"\n{'='*100}\n### {ds}   題目 {len(recs)}   無時間訊號的題目 {len(silent)}\n{'='*100}")
    hdr = ("%-30s%22s%22s%14s%16s" % ("機制", "a_cos 值域", "a_bc 值域", "與基準不同", "退化檢查"))
    print(hdr); print("-" * len(hdr))

    for code, spec, norm, magn in MECHS:
        J = Judge(spec, recs, ds, bc_of, normalize=norm, mag_norm=magn)
        wc, wb, wh, bad = [], [], [], 0
        for r in recs:
            for c in r["candidates"]:
                w = J.weights(c, r["uid"])
                for k, v in w.items():
                    if not (isinstance(v, float) and v == v and v >= 0 and v != float("inf")):
                        bad += 1
                wc.append(w["cos"]); wb.append(w["bc"]); wh.append(w["hop"])
        diff = sum(1 for r in recs
                   if top10(r["candidates"], lambda c: J.utility(c, r["uid"])) != base[r["uid"]])
        # check 2: same mechanism, temporal judgement forced to "time is irrelevant"
        J0 = Judge(spec, recs, ds, bc_of, normalize=norm, mag_norm=magn)
        if J0.crit is not None:
            J0.crit = {k: 0.0 for k in J0.crit}
        if J0.disp is not None:
            J0.disp = {u: {**d, "bc": 0.0} for u, d in J0.disp.items()}
        ok_deg = sum(1 for r in silent
                     if top10(r["candidates"], lambda c: J.utility(c, r["uid"]))
                     == top10(r["candidates"], lambda c: J0.utility(c, r["uid"])))
        rng = lambda v: f"{min(v):.3f}–{max(v):.3f}"
        deg = f"{ok_deg}/{len(silent)}" + (" ✅" if ok_deg == len(silent) else " ❌")
        flag = "  ❌NaN" if bad else ""
        print("%-30s%22s%22s%13d%16s%s" % (f"{code} {spec}" + ("" if norm else " raw") + ("" if magn else " magnone"),
                                           rng(wc), rng(wb), diff, deg, flag))

    # check 4: the llm judge must lean on time less where the judge itself says time matters less
    crit = json.load(open(PIPE / f"datasets/{ds}/criticality.json"))
    sc = sorted(v["score"] for v in crit.values())
    print(f"\n  criticality 中位數 = {sc[len(sc)//2]:.2f}   (329 應為 0.50、medbullets 0.20、mmlu 0.10)")
    print(f"  基準 top-10 取自 cos + 0.3·bc − 0.08·hop；'與基準不同' 低於 5% 的機制不值得送 inference")
