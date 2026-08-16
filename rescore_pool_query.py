#!/usr/bin/env python3
"""Re-score a pool's cos against a richer query, without changing which candidates are in it.

WHY. cos is currently the similarity to the SYMPTOM phrases alone (kg_walker.encode_query over
symptoms.json). That is why disease names lose to symptom synonyms: a disease name is structurally
less similar to symptom text than a reworded symptom is. Measured on the union pool, raw_1hop's
candidates are 3.2x more likely to be gold than the walker's while scoring 0.15 LOWER on cos, and
80% of them fall below the walk's own tau. No utility that increases in cos can prefer them.

query_entities.json carries 13 phrases per question against symptoms.json's 6 -- diseases mentioned,
procedures, drugs, lab findings -- so the query can describe the case rather than only its
complaints. Changing the query changes every candidate's score but NOT the candidate set, so this
stays a pool-level replay: no Neo4j, no LLM.

KNOWN RISK, recorded rather than discovered later: `diseases_mentioned` includes past history, and
an earlier attempt at a wider query was abandoned because past-history diseases hijacked the
ranking. QUERY=all keeps that risk; QUERY=sym+dx and QUERY=sym+lab isolate it.

Usage:
  DATASET=medbullets METHOD=union QUERY=all python3 pipeline/rescore_pool_query.py
"""
import json, os, pickle, sys
import numpy as np
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "code")); sys.path.insert(0, str(PIPE / "code" / "dkr_policy"))
os.environ.setdefault("DKR_MATRIX_PKL", str(PIPE / "cache/umls_broad_embeddings_sapbert.pkl"))
DS = os.environ.get("DATASET", "medbullets")
METHOD = os.environ.get("METHOD", "union")
QUERY = os.environ.get("QUERY", "all")

FIELDS = {"all":    ["symptoms_signs", "diseases_mentioned", "procedures", "drugs", "lab_findings"],
          "sym+dx": ["symptoms_signs", "diseases_mentioned"],
          "sym+lab": ["symptoms_signs", "lab_findings"]}[QUERY]

idx = pickle.load(open(os.environ["DKR_MATRIX_PKL"], "rb"))
cui2row = {c: i for i, c in enumerate(idx["cuis"])}
M = idx["matrix"]
from kg_walker import encode_query
sym = json.load(open(PIPE / f"datasets/{DS}/symptoms.json"))
qe = json.load(open(PIPE / f"datasets/{DS}/query_entities.json"))

out = PIPE / f"pool/{DS}/{METHOD}_q{QUERY.replace('+','')}.jsonl"
n_ph, changed, kept = [], 0, 0
with open(out, "w") as fh:
    for line in open(PIPE / f"pool/{DS}/{METHOD}.jsonl"):
        r = json.loads(line)
        u = r["uid"]
        ph = list((sym.get(u) or {}).get("symptoms") or [])
        for f in FIELDS:
            ph += [x for x in ((qe.get(u) or {}).get(f) or []) if isinstance(x, str)]
        ph = [p for p in dict.fromkeys(ph) if p.strip()]
        if not ph or not r["candidates"]:
            fh.write(line); continue
        n_ph.append(len(ph))
        q = np.asarray(encode_query(ph), dtype=M.dtype); q /= (np.linalg.norm(q) or 1.0)
        for c in r["candidates"]:
            row = cui2row.get(c.get("cui"))
            if row is None:
                kept += 1; continue          # no embedding row: keep the stored cos
            new = float(M[row] @ q)
            if abs(new - c.get("cos", 0.0)) > 1e-6: changed += 1
            c["cos"] = new
            c["score"] = new + 0.3 * c.get("bc", 0.0) - 0.08 * c.get("hop", 0)
        r["candidates"].sort(key=lambda c: -c["score"])
        for i, c in enumerate(r["candidates"], 1): c["rank"] = i
        r["params"] = {**(r.get("params") or {}), "query": QUERY, "query_fields": FIELDS}
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{DS}/{METHOD} QUERY={QUERY} -> {out.name}")
print(f"  query 片語數 中位 {sorted(n_ph)[len(n_ph)//2]}（原本只有 symptoms）"
      f"  重算 {changed} 個候選，無 embedding 而保留原值 {kept} 個")
