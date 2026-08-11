#!/usr/bin/env python3
"""#8 (less-misleading story): 'retrieves less, but less misleading', across ALL retrieval methods.

For each method, at a FIXED budget (top-10 retrieved items), a single LLM-judge call returns
which OPTION letters are present/supported in the retrieved evidence. From that:

  Recall@10            = gold option present in top-10
  Distractor contam.   = mean fraction of WRONG options also present in top-10
  ---- retrieval-attributable error (two definitions) ----
  RAE_simple  = P( predicted-option present in retrieved | answer is WRONG )
                i.e. of all wrong answers, how many picked an option the retrieval had surfaced.
  RAE_causal  = P( predicted-option present AND no-retrieval CoT was CORRECT | answer is WRONG )
                i.e. errors the retrieval plausibly INTRODUCED: the reader would have been right
                without retrieval, but retrieval surfaced the distractor it then followed.
  P(correct|gold in) / P(correct|gold out)  = reader accuracy split by whether gold was surfaced.

Same judge, same top-10 budget across methods -> fair. gpt-5.4-mini, bench329 (n=329).
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; CACHE = ROOT/"cache"
sys.path.insert(0, str(ROOT/"scripts"))
from spectrum_textbook import call_llm

B = {x["uid"]: x for x in json.load(open(CACHE/"benchmark_bench329_clean.json"))}
# no-retrieval reference for the CAUSAL attribution: did plain CoT get it right?
COT = {x["uid"]: bool(x.get("is_correct"))
       for x in json.load(open(CACHE/"rerun329_cot_minimal__gpt-5.4-mini.json"))["results"]}

NUM = re.compile(r"^\s*\d+\.\s+(.*\S)\s*$")
def concepts_from_block(blk, k=10):
    out=[]
    for ln in (blk or "").splitlines():
        m=NUM.match(ln)
        if m: out.append(re.sub(r"\s*\([^)]*\)\s*$","",m.group(1)).strip())
        if len(out)>=k: break
    return out

def medrag_excerpts(prompt, k=10):
    j=prompt.find("Retrieved textbook")
    blob = prompt[j:] if j>=0 else prompt
    parts=re.split(r"\[Excerpt \d+\][^\n]*", blob)[1:]
    return [p.strip().replace("\n"," ")[:400] for p in parts][:k]

JUDGE=('Retrieved evidence for a clinical case (top items):\n{lst}\n\n'
 'Answer options:\n{opts}\n\n'
 'For EACH option letter, decide if THAT diagnosis is present/supported in the retrieved evidence '
 '- by name, or as a clear clinical synonym / subtype / more-specific or more-general form of the '
 'SAME disease entity. A different disease in the same broad family does NOT count.\n'
 'Reply with ONLY the letters that ARE present, comma-separated (e.g. "A,C"), or "NONE".')

def judge_present(items, opts):
    if not items: return set()
    ol="\n".join(f"  {k}. {v}" for k,v in sorted(opts.items()))
    r=(call_llm(JUDGE.format(lst="\n".join(f"- {c}" for c in items), opts=ol),
                model="gpt-5.4-mini") or "").upper()
    return set(re.findall(r"[A-J]", r))

def load(f):
    d=json.load(open(CACHE/f)); return {x["uid"]:x for x in (d["results"] if "results" in d else d)}

# method -> (file, extractor)
def blk_extractor(field):
    return lambda rec: concepts_from_block(rec.get(field), 10)
METHODS = {
  "walker(cos+bc)": ("bench340_walker_fixABCmtS__gpt-5.4-mini.json", blk_extractor("kg_block")),
  "raw_1hop_dump":  ("bench329_raw_1hop_K500__gpt-5.4-mini.json",     blk_extractor("kg_block")),
  "raw_2hop_dump":  ("bench329_raw_2hop_K500__gpt-5.4-mini.json",     blk_extractor("kg_block")),
  "ToG":            ("tog_baseline_329__gpt-5.4-mini.json",           blk_extractor("kg_block")),
  "HyKGE":          ("hykge_baseline_329__gpt-5.4-mini.json",         blk_extractor("kg_block")),
  "MedRAG":         ("rerun329_MedRAG__gpt-5.4-mini.json",            lambda rec: medrag_excerpts(rec.get("prompt_full",""), 10)),
}

results={}; summary=[]
for name,(f,extract) in METHODS.items():
    R=load(f); uids=[u for u in B if u in R]
    rows=[]
    for i,u in enumerate(uids):
        opts=B[u]["options"]; gold=B[u]["answer"]
        pred=R[u].get("predicted"); corr=bool(R[u].get("is_correct"))
        present=judge_present(extract(R[u]), opts)
        wrongs=[o for o in opts if o!=gold]
        rows.append({"uid":u,"gold":gold,"pred":pred,"correct":corr,
                     "gold_in":gold in present,
                     "distractors_in":sum(1 for w in wrongs if w in present),
                     "n_distractors":len(wrongs),
                     "pred_in":(pred in present) if pred else False,
                     "cot_correct":COT.get(u)})
        if (i+1)%80==0: print(f"  {name} ...{i+1}/{len(uids)}",flush=True)
    results[name]=rows; n=len(rows)
    recall =100*sum(r["gold_in"] for r in rows)/n
    contam =100*sum(r["distractors_in"]/max(1,r["n_distractors"]) for r in rows)/n
    wrong=[r for r in rows if not r["correct"]]
    rae_s =100*sum(r["pred_in"] for r in wrong)/max(1,len(wrong))
    rae_c =100*sum(r["pred_in"] and r["cot_correct"] for r in wrong)/max(1,len(wrong))
    gi=[r for r in rows if r["gold_in"]]; go=[r for r in rows if not r["gold_in"]]
    p_in =100*sum(r["correct"] for r in gi)/max(1,len(gi))
    p_out=100*sum(r["correct"] for r in go)/max(1,len(go))
    summary.append((name,recall,contam,rae_s,rae_c,p_in,p_out,len(wrong)))
    json.dump(results, open(CACHE/"misleading_metrics_329.json","w"), indent=1)
    print(f"  [done] {name}", flush=True)

print("\n"+"="*104)
print(f"{'method':16s} {'Rec@10':>7s} {'Contam':>7s} {'RAE_simple':>11s} {'RAE_causal':>11s} {'P(c|gold_in)':>13s} {'P(c|gold_out)':>14s} {'nWrong':>7s}")
for name,rec,con,rs,rc,pi,po,nw in summary:
    print(f"{name:16s} {rec:6.1f}% {con:6.1f}% {rs:10.1f}% {rc:10.1f}% {pi:12.1f}% {po:13.1f}% {nw:7d}")
json.dump({"summary":[dict(zip(['method','recall','contam','rae_simple','rae_causal','p_correct_gold_in','p_correct_gold_out','n_wrong'],s)) for s in summary]},
          open(CACHE/"misleading_metrics_summary_329.json","w"), indent=2)
print("\nsaved -> cache/misleading_metrics_329.json + _summary_329.json")
