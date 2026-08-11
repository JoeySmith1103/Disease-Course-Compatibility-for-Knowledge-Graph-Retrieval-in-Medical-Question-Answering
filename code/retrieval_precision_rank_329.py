#!/usr/bin/env python3
"""#8 (intrinsic retrieval quality): directly measure 'retrieves less but more relevant,
ranked higher, less noise' on the retrieved SET itself (before the reader).

Per (question, method): take the top-10 retrieved concepts WITH their list positions, and one
LLM-judge call marks which are clinically RELEVANT to THIS case (a plausible differential dx or a
directly diagnostic finding/cause) vs generic / navigational / off-topic concepts.

  Precision@10   = |relevant| / 10                 (relevance density; higher = cleaner)
  Noise ratio    = 1 - Precision@10                 (off-topic share; lower = cleaner)
  MeanRank(rel)  = mean list-position of relevant   (lower = relevant items sit earlier)
  Rank1(rel)     = position of first relevant        (lower = a relevant item appears sooner)

Same judge, same top-10 budget across methods -> fair. gpt-5.4-mini, bench329 (n=329).
KG-concept methods only (MedRAG is text excerpts, excluded).
"""
import json, re, sys, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; CACHE = ROOT/"cache"
sys.path.insert(0, str(ROOT/"scripts"))
from spectrum_textbook import call_llm

B = {x["uid"]: x for x in json.load(open(CACHE/"benchmark_bench329_clean.json"))}
NUM = re.compile(r"^\s*\d+\.\s+(.*\S)\s*$")
def concepts(blk, k=10):
    out=[]
    for ln in (blk or "").splitlines():
        m=NUM.match(ln)
        if m: out.append(re.sub(r"\s*\([^)]*\)\s*$","",m.group(1)).strip())
        if len(out)>=k: break
    return out

JUDGE=('Clinical case:\n{q}\n\nRetrieved candidate concepts (numbered):\n{lst}\n\n'
 'For THIS specific case, which numbered concepts are clinically RELEVANT — i.e. a plausible '
 'differential diagnosis, or a directly diagnostic finding / cause / complication for this '
 'presentation? EXCLUDE generic or navigational concepts (e.g. "Disease", "Finding", "Infectious '
 'agent", a bare anatomy term, or an organism/term unrelated to this patient).\n'
 'Reply with ONLY the relevant numbers, comma-separated (e.g. "1,3,4"), or "NONE".')

def judge_relevant(q, cands):
    if not cands: return set()
    lst="\n".join(f"{i}. {c}" for i,c in enumerate(cands,1))
    # full case text — the old q[:1400] cut off 11/329 vignettes (longest 1877 chars), so the
    # judge was rating relevance without having seen the end of those cases
    r=(call_llm(JUDGE.format(q=q, lst=lst), model="gpt-5.4-mini") or "")
    return {int(x) for x in re.findall(r"\b([1-9]|10)\b", r)}

def load(f):
    d=json.load(open(CACHE/f)); return {x["uid"]:x for x in (d["results"] if "results" in d else d)}

METHODS = {
  "walker(cos+bc)": "bench340_walker_fixABCmtS__gpt-5.4-mini.json",
  "raw_1hop_dump":  "bench329_raw_1hop_K500__gpt-5.4-mini.json",
  "raw_2hop_dump":  "bench329_raw_2hop_K500__gpt-5.4-mini.json",
  "ToG":            "tog_baseline_329__gpt-5.4-mini.json",
  "HyKGE":          "hykge_baseline_329__gpt-5.4-mini.json",
}

results={}; summary=[]
for name,f in METHODS.items():
    R=load(f); uids=[u for u in B if u in R]
    rows=[]
    for i,u in enumerate(uids):
        cands=concepts(R[u].get("kg_block"), 10)
        if not cands: continue
        rel=judge_relevant(B[u]["question"], cands)
        rel={x for x in rel if 1<=x<=len(cands)}
        rows.append({"uid":u,"n_cands":len(cands),"rel_positions":sorted(rel),
                     "precision":len(rel)/len(cands),
                     "mean_rank":(statistics.fmean(rel) if rel else None),
                     "rank1":(min(rel) if rel else None)})
        if (i+1)%80==0: print(f"  {name} ...{i+1}/{len(uids)}",flush=True)
    results[name]=rows
    n=len(rows)
    prec=100*statistics.fmean(r["precision"] for r in rows)
    noise=100-prec
    mr=[r["mean_rank"] for r in rows if r["mean_rank"] is not None]
    r1=[r["rank1"] for r in rows if r["rank1"] is not None]
    meanrank=statistics.fmean(mr) if mr else float('nan')
    rank1=statistics.fmean(r1) if r1 else float('nan')
    frac_norel=100*sum(1 for r in rows if not r["rel_positions"])/n
    summary.append((name,n,prec,noise,meanrank,rank1,frac_norel))
    json.dump(results, open(CACHE/"retrieval_precision_rank_329.json","w"), indent=1)
    print(f"  [done] {name}: prec={prec:.1f}% meanrank={meanrank:.2f}", flush=True)

print("\n"+"="*92)
print(f"{'method':16s} {'n':>4s} {'Prec@10':>8s} {'Noise':>7s} {'MeanRank(rel)':>14s} {'Rank1(rel)':>11s} {'%no-rel':>8s}")
for name,n,prec,noise,meanrank,rank1,fnr in summary:
    print(f"{name:16s} {n:4d} {prec:7.1f}% {noise:6.1f}% {meanrank:13.2f} {rank1:10.2f} {fnr:7.1f}%")
json.dump({"summary":[dict(zip(['method','n','precision','noise','mean_rank_relevant','rank1_relevant','pct_no_relevant'],s)) for s in summary]},
          open(CACHE/"retrieval_precision_rank_summary_329.json","w"), indent=2)
print("\nsaved -> cache/retrieval_precision_rank_329.json + _summary_329.json")
