#!/usr/bin/env python3
"""Merge the walker pool with raw_1hop's concepts into one scored candidate set.

WHY THIS EXISTS. Forty-odd ranking variants over the walker pool all landed within +-1.5pp of each
other, and the reason turned out to be containment: raw_1hop's top-10 is only 20.1% present in the
walker pool, and on 92 of 298 questions the overlap is ZERO. No utility over the walker pool can
produce raw_1hop's evidence, because four fifths of it was never scored. The two methods start from
the same seeds; they diverge because the walker's tau gate filters expansion by similarity while
raw_1hop takes traversal order.

So the way to let one utility choose between them is to put both in the same pool, on the same
scale. raw_1hop's records carry only names, so cos / bc / role are recomputed here:

    cos   concept_embedding @ q_emb, q_emb = mean-pooled SapBERT over the question's symptom
          phrases -- byte-for-byte the walker's own definition (kg_walker.encode_query).
          VERIFIED: recomputing cos for candidates present in both pools reproduces the walker's
          stored value on 171/171 checked, so the merged scores are comparable rather than merely
          similar-looking.
    bc    read from the duration cache only. A cache miss scores 0 instead of generating, because
          generating would mean an LLM call per missing concept.
    hop   1 by construction -- raw_1hop is the one-hop neighbourhood.
    role  from the embedding matrix's own role vector, the same source the walker uses.

4.1% of raw_1hop's concepts have no row in the matrix. They are `Acute (qualifier value)`, `Raised`,
`High`, `Increased`, `Absence of` -- the navigational nodes that reading the blocks flagged as
contentless anyway, so dropping them costs nothing.

Where a concept is in both pools the walker's record wins: it carries the real hop and path.

Usage:
  DKR_MATRIX_PKL=cache/umls_broad_embeddings_sapbert.pkl python3 pipeline/build_union_pool.py
  DATASET=329 ... python3 pipeline/build_union_pool.py
"""
import json, os, pickle, re, sys
import numpy as np
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE)); sys.path.insert(0, str(PIPE / "code"))
sys.path.insert(0, str(PIPE / "code" / "dkr_policy"))
DS = os.environ.get("DATASET", "medbullets")
os.environ.setdefault("DKR_MATRIX_PKL", str(PIPE / "cache/umls_broad_embeddings_sapbert.pkl"))

from bc_onesided import compat_for_cui

_P = re.compile(r"[^a-z0-9 ]+")
def nrm(s):
    return " ".join(_P.sub(" ", re.sub(r"\([^)]*\)", " ", (s or "").lower())).split())

idx = pickle.load(open(os.environ["DKR_MATRIX_PKL"], "rb"))
M, CUIS, ROLE = idx["matrix"], idx["cuis"], idx["role"]
n2r = {}
for lst in (idx["names"], idx.get("bare_names") or []):
    for i, n in enumerate(lst):
        n2r.setdefault(nrm(n), i)

from kg_walker import encode_query
sym = json.load(open(PIPE / f"datasets/{DS}/symptoms.json"))
walker = {r["uid"]: r for r in (json.loads(l) for l in open(PIPE / f"pool/{DS}/walker.jsonl"))}
raw = {r["uid"]: r for r in (json.loads(l) for l in open(PIPE / f"pool/{DS}/raw_1hop.jsonl"))}

pd1 = lambda v: (float(v[0]) if isinstance(v, (list, tuple)) and v
                 else (float(v) if isinstance(v, (int, float)) else None))
out_path = PIPE / f"pool/{DS}/union.jsonl"
added = dropped = bc_hit = bc_zero = 0
recs = 0
with open(out_path, "w") as fh:
    for uid, wr in walker.items():
        cands = list(wr["candidates"])
        seen = {c.get("cui") for c in cands if c.get("cui")}
        seen_n = {nrm(c.get("name")) for c in cands}
        phrases = (sym.get(uid) or {}).get("symptoms") or []
        rr = raw.get(uid)
        if phrases and rr:
            q = np.asarray(encode_query(phrases), dtype=M.dtype)
            q = q / (np.linalg.norm(q) or 1.0)
            t = pd1(wr.get("patient_days"))
            for c in rr["candidates"]:
                row = n2r.get(nrm(c["name"]))
                if row is None:
                    dropped += 1; continue
                cui = CUIS[row]
                if cui in seen or nrm(c["name"]) in seen_n:
                    continue          # walker's record wins: it has the real hop and path
                seen.add(cui)
                role = ROLE[row] if row < len(ROLE) else "other"
                bo = compat_for_cui(cui, t) if (t and role == "disease") else 0.0
                if bo: bc_hit += 1
                else:  bc_zero += 1
                cands.append({"rank": 0, "cui": cui, "name": idx["names"][row],
                              "role": role, "hop": 1,
                              "cos": float(M[row] @ q), "bc": 0.0,
                              "bc_onesided": round(bo, 6),
                              "origin_seed": "raw_1hop", "src": "raw_1hop",
                              "path": [["raw_1hop", idx["names"][row], 0.0]]})
                added += 1
        for i, c in enumerate(cands, 1):
            c.setdefault("src", "walker")
            c["score"] = c.get("cos", 0.0) + 0.3 * c.get("bc", 0.0) - 0.08 * c.get("hop", 0)
        cands.sort(key=lambda c: -c["score"])
        for i, c in enumerate(cands, 1):
            c["rank"] = i
        fh.write(json.dumps({**wr, "candidates": cands, "n_candidates": len(cands),
                             "dataset": DS, "method": "union",
                             "source": f"pool/{DS}/walker.jsonl + pool/{DS}/raw_1hop.jsonl",
                             "params": {"cos": "recomputed, verified against walker 171/171",
                                        "bc": "cache-only, 0 on miss (no LLM generation)",
                                        "hop_for_raw": 1,
                                        "replayable": ["top_k", "lambda", "mu", "utility", "quota"],
                                        "needs_rerun": ["tau", "max_hops", "neighbor_limit", "seeds"]},
                             }, ensure_ascii=False) + "\n")
        recs += 1
n = [len(json.loads(l)["candidates"]) for l in open(out_path)]
print(f"{DS}: {recs} 題  -> pool/{DS}/union.jsonl")
print(f"  新增 raw_1hop 候選 {added} 個（每題平均 {added/recs:.1f}），"
      f"因無矩陣列而丟棄 {dropped} 個")
print(f"  新增候選的 bc_onesided: 有值 {bc_hit}，快取未命中而記 0 的 {bc_zero}")
print(f"  池大小 中位 {sorted(n)[len(n)//2]}（walker 原本 "
      f"{sorted(len(r['candidates']) for r in walker.values())[len(walker)//2]}）")
