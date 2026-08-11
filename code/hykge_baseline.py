#!/usr/bin/env python3
"""Faithful HyKGE baseline on our UMLS/SNOMED Neo4j KG.

Mirrors Artessay/HyKGE (HO -> NER -> KG path retrieval -> HOFGR rerank -> answer):
  1. HO (Hypothesis Output): LLM writes a hypothetical answer to the question (HyDE),
     plus the key medical entities it mentions (one call).
  2. NER / anchors: anchor entities = question seeds (LLM-DDx) UNION HO entities, each
     linked to a UMLS CUI by SapBERT nearest-neighbor.
  3. KG retrieval: for every ANCHOR PAIR, collect all THREE chain types the paper uses —
     (a) direct path via shortestPath (<= MAX_HOP=3, the paper's k),
     (b) CONVERGING "co-ancestor" chain  (ci -> m <- cj),
     (c) DIVERGING "co-occurrence" chain (ci <- m -> cj);
     plus 1-hop neighbors of anchors. Collect chain entities.
  4. HOFGR (Filter): rerank retrieved chain fragments against (question+HO); keep TOP_SIM=10.
  5. Answer: LLM answers the MCQ from the reranked chains.
Records explored chain entities (recall@10) and the final answer (accuracy).

FIDELITY vs the paper (arXiv 2312.15883) — "HyKGE + our settings".
Audited step-by-step against the paper's full pipeline, not spot-checked.

  MATCHED:
    * HO hypothesis step, with the paper's medical-expert + chain-of-thought framing and its
      decoding settings (temperature=0.6, max_tokens=300).
    * Anchors drawn from BOTH the question and the HO.
    * All THREE chain types (direct / co-ancestor / co-occurrence), k=3.
    * HOFGR *fragment granularity*: [Q||HO] chunked into overlapping windows (lc=10, oc=4) and
      each path scored against its BEST-matching fragment (max over chunks).
    * topK=10; single-pass retrieval; no entity-type filtering; no self-consistency — as in the paper.
    * Final answer prompt = query + pruned paths, and deliberately NOT the HO text.

  SUBSTITUTED (the paper's components are Chinese-specific and cannot run on an English UMLS KG):
    * anchor NER + linking: W2NER + GTE encoder (delta=0.7)  ->  our shared LLM-DDx seed set
      + HO entities, SapBERT-linked (0.5). Using the SAME seeds as every other method here is
      also a deliberate controlled-comparison choice.
    * HOFGR scorer: trained cross-encoder `bge_reranker_large`  ->  SapBERT bi-encoder cosine.
      NOTE: only the SCORER differs now; the fragment-granularity half of HOFGR is implemented.

  UNAVAILABLE IN OUR KG:
    * The paper renders each path with its source/target entity DESCRIPTIONS. Our Neo4j Concept
      nodes carry only {CUI, name} — there is no description property to render.

  REMOVED (was an extra of ours, NOT in the paper):
    * raw 1-hop neighbours of every anchor were being added to the candidate pool. Now opt-in
      via HYKGE_ADD_1HOP=1, default off.
"""
import json, os, re, sys, time, itertools, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; CACHE = ROOT/"cache"
# ROOT is pipeline/; the modules live in pipeline/code and pipeline/code/dkr_policy
sys.path.insert(0, str(ROOT/"code")); sys.path.insert(0, str(ROOT/"code"/"dkr_policy"))
os.environ.setdefault("DKR_MATRIX_PKL", "cache/umls_broad_embeddings_sapbert.pkl")

MAX_HOP = int(os.environ.get("HYKGE_HOP", "3"))   # paper: k=3 max hops
TOP_SIM = int(os.environ.get("HYKGE_TOPK", "10"))
N_ANCHORS = int(os.environ.get("HYKGE_ANCHORS", "10"))

B = {x["uid"]: x for x in json.load(open(os.environ.get("BENCH_PATH", str(CACHE/"benchmark_bench329_clean.json"))))}
seeds_c = json.load(open(os.environ.get("SEEDS_PATH", str(CACHE/"bench340_seeds_multitype.json"))))
sym_c   = json.load(open(os.environ.get("SYM_PATH",   str(CACHE/"bench340_symptoms.json"))))

HO_PROMPT = """### Task Description:
You are a medical expert. Please write a passage to answer [User Query] while adhering to [Answer Requirements].
### Answer Requirements:
1) Please take time to think slowly, understand step by step, and answer questions. Do not skip key steps.
2) Fully analyze the problem through thinking and exploratory analysis.
### User Query
{q}

(Finally, on a last line starting with "ENTITIES:", list the key medical entities you mentioned,
comma-separated. This line stands in for the paper's separate W2NER pass over [Q||HO].)"""

# Paper decoding settings for the hypothesis step.
HO_TEMPERATURE = 0.6
HO_MAX_TOKENS = 300

# HOFGR "fragment granularity": [Q||HO] is split into overlapping windows and each path is
# scored against the BEST-matching fragment, not against one averaged vector.
CHUNK_LC = int(os.environ.get("HYKGE_CHUNK_LC", "10"))   # window size (paper: 10)
CHUNK_OC = int(os.environ.get("HYKGE_CHUNK_OC", "4"))    # overlap    (paper: 4)
# The paper does NOT add raw 1-hop neighbours of anchors; that was an extra of ours.
ADD_1HOP = os.environ.get("HYKGE_ADD_1HOP", "0") == "1"


def chunk_text(text, lc=None, oc=None):
    """Overlapping word windows over [Q||HO] (HOFGR fragment granularity)."""
    lc = lc or CHUNK_LC; oc = oc or CHUNK_OC
    w = (text or "").split()
    if len(w) <= lc: return [text or ""]
    step = max(1, lc - oc)
    return [" ".join(w[i:i+lc]) for i in range(0, len(w) - oc, step)] or [text]

ANSWER = """### Task Description:
You are a medical expert. Based on relevant medical [Background Knowledge] and your medical knowledge,
answer the [User Query] while adhering to [Answer Requirements].
### Answer Requirements:
1) Take time to think slowly, understand step by step, and answer questions.
2) Clearly state key information in the answer and provide direct and specific answers to user questions.
### Background Knowledge
The retrieved knowledge chains are:
{chains}
### User Query
{q}
Options:
{opts}
Reason briefly (<150 words), then give the final answer as a single letter in <a></a> tags."""

def extract_letter(raw):
    if not raw: return None
    for pat in [r"<a>\s*([A-J])\s*</a>", r"\\boxed\{\s*([A-J])\s*\}",
                r"\banswer\s*(?:is|:)\s*\**\(?([A-J])\)?\b"]:
        m=re.search(pat, raw, re.IGNORECASE)
        if m: return m.group(1).upper()
    return None

_G={}
def _load():
    if _G: return
    from duration_kg_rag import _load_umls_emb
    idx,enc=_load_umls_emb()           # single SapBERT matrix load (DKR_MATRIX_PKL)
    mat=idx["matrix"]; mat=mat/np.maximum(np.linalg.norm(mat,axis=1,keepdims=True),1e-12)
    _G["mat"]=mat; _G["cuis"]=idx["cuis"]; _G["cui2idx"]={c:i for i,c in enumerate(idx["cuis"])}; _G["enc"]=enc
    from umls_neo4j import get_driver; _G["drv"]=get_driver()
    # Use the SAME relation blocklist as the walker so both methods see the same graph.
    # SNOMED context wrappers (has_finding_context -> "Known present (qualifier value)")
    # are structurally valid but clinically vacuous; CMeKG has no equivalent, so their
    # code needs no such filter. Filtering only one method would handicap the baseline.
    from kg_walker import _BAD_RELA; _G["bad"]=list(_BAD_RELA)

def emb(texts):
    texts=[str(t or "x") for t in texts]
    v=_G["enc"].encode(texts); return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)

def link(names):
    """nearest-neighbor CUI link for each name (>0.5)."""
    if not names: return []
    e=emb(names); out=[]
    for i in range(len(names)):
        sc=_G["mat"]@e[i]; j=int(np.argmax(sc))
        if sc[j]>0.5: out.append((_G["cuis"][j], _G["cuis"][j]))
    # real names
    res=[]; seen=set()
    with _G["drv"].session() as s:
        for cui,_ in out:
            if cui in seen: continue
            seen.add(cui)
            r=s.run("MATCH (c:Concept {CUI:$c}) RETURN c.name AS n LIMIT 1", c=cui).single()
            res.append((cui, r["n"] if r else cui))
    return res

def direct_edge(ci, cj):
    """Their PathExplorer case 1: a DIRECT edge between two anchors (path=[src,dst], 1 hop).
    Their adjacency list is undirected, so orientation is recovered for rendering."""
    with _G["drv"].session() as s:
        r = s.run("MATCH (a:Concept {CUI:$ci})-[r]-(b:Concept {CUI:$cj}) "
                  "WHERE r.RELA IS NOT NULL AND NOT r.RELA IN $bad "
                  "RETURN r.RELA AS rel, (startNode(r) = a) AS fwd LIMIT 1",
                  ci=ci, cj=cj, bad=_G["bad"]).single()
    return r


def shared_intermediate(ci, cj):
    """Their PathExplorer case 2: path=[src, mid, dst] through ONE shared neighbour, on an
    UNDIRECTED adjacency list. Their three "chain types" are not three queries — they are a
    post-hoc shape test on this same path (KGModule: in_S+out_S == 2 or == 0), i.e.

        a->m->b  straight | a->m<-b  co-ancestor | a<-m->b  co-occurrence

    so one undirected query yields all three. `max_hop` in their config is dead code; the real
    limit is 2 hops, which is why 1 hop per side is the correct granularity."""
    with _G["drv"].session() as s:
        r = s.run("MATCH (a:Concept {CUI:$ci})-[r1]-(m:Concept)-[r2]-(b:Concept {CUI:$cj}) "
                  "WHERE a<>b AND m<>a AND m<>b "
                  "AND r1.RELA IS NOT NULL AND r2.RELA IS NOT NULL "
                  "AND NOT r1.RELA IN $bad AND NOT r2.RELA IN $bad "
                  "RETURN m.CUI AS mcui, m.name AS mid, r1.RELA AS rel1, r2.RELA AS rel2, "
                  "(startNode(r1) = a) AS a_to_m, (startNode(r2) = m) AS m_to_b LIMIT 1",
                  ci=ci, cj=cj, bad=_G["bad"]).single()
    return r


def _arrow(names, rels, forward=True):
    """Render a path in the paper's arrow style: `A →rel→ B →rel→ C`."""
    parts=[names[0]]
    for i,r in enumerate(rels or []):
        parts.append(f"→{r}→" if forward else f"←{r}←")
        if i+1 < len(names): parts.append(names[i+1])
    return " ".join(parts)


def _join_at_hub(left, right):
    """Splice two renderings that share their LAST/FIRST node (the hub) without repeating it."""
    return left + " " + " ".join(right.split(" ")[1:])


def one_hop(cui):
    with _G["drv"].session() as s:
        rows=s.run("MATCH (a:Concept {CUI:$c})-[r]-(b:Concept) WHERE r.RELA IS NOT NULL "
                   "RETURN b.CUI AS cui, b.name AS name, r.RELA AS rela LIMIT 80", c=cui).data()
    return [(x["cui"],x["name"],x["rela"]) for x in rows if x["cui"]]

def process_one(item):
    from spectrum_textbook import call_llm
    _load()
    uid=item["uid"]; gold=item["answer"]; opts=item["options"]; q=item["question"]
    # 1. HO + entities
    ho_raw=call_llm(HO_PROMPT.format(q=q), model=os.environ.get("HYKGE_LLM","gpt-5.4-mini"),
                    temperature=HO_TEMPERATURE, max_tokens=HO_MAX_TOKENS) or ""
    ho_text=ho_raw; ho_ents=[]
    m=re.search(r"ENTITIES:\s*(.+)", ho_raw, re.IGNORECASE|re.DOTALL)
    if m:
        ho_text=ho_raw[:m.start()].strip()
        ho_ents=[e.strip() for e in re.split(r"[,;\n]", m.group(1)) if e.strip()][:10]
    # 2. anchors = question seeds UNION HO entities
    # Their NER runs over [Q || HO], so the query half must be the query's OWN findings, not
    # LLM-DDx. symptoms.json holds the observed findings; seeds.json is already hypothesis-like.
    qsym=(sym_c.get(uid) or {}).get("symptoms",[])[:6]
    qseeds=(seeds_c.get(uid) or {}).get("seeds",[])[:4]
    anchors=link(list(dict.fromkeys(qsym+qseeds+ho_ents)))[:N_ANCHORS]
    acuis=[c for c,_ in anchors]
    # 3. KG retrieval: chains between anchor pairs + 1-hop
    chain_nodes={}  # cui -> name
    chain_strs=[]
    names={c:n for c,n in anchors}
    for ci,cj in itertools.combinations(acuis, 2):
        d = direct_edge(ci, cj)
        if d:
            a,b = names.get(ci,ci), names.get(cj,cj)
            chain_strs.append(f"{a} →{d['rel']}→ {b}" if d["fwd"] else f"{a} ←{d['rel']}← {b}")
        m = shared_intermediate(ci, cj)
        if m:
            chain_nodes.setdefault(m["mcui"], m["mid"])
            a,b,mid = names.get(ci,ci), names.get(cj,cj), m["mid"]
            l = f"{a} →{m['rel1']}→ {mid}" if m["a_to_m"] else f"{a} ←{m['rel1']}← {mid}"
            r = f" →{m['rel2']}→ {b}" if m["m_to_b"] else f" ←{m['rel2']}← {b}"
            chain_strs.append(l + r)
    if ADD_1HOP:   # not part of HyKGE; kept only as an opt-in extra
        for c in acuis:
            for ncui,nname,rela in one_hop(c):
                chain_nodes.setdefault(ncui,nname)
    # 4. HOFGR rerank — FRAGMENT granularity: chunk [Q||HO] into overlapping windows and score
    #    each candidate against its BEST-matching fragment (max over chunks), not against one
    #    averaged vector. Only the scorer itself still differs from the paper (SapBERT
    #    bi-encoder cosine instead of the bge_reranker_large cross-encoder).
    frags = chunk_text(q + " " + ho_text)
    FE = emb(frags)                                   # (n_frag, d)
    cand=[(cu,nm) for cu,nm in chain_nodes.items() if cu not in set(acuis)]
    if cand:
        ce=emb([nm for _,nm in cand])
        sims=(ce@FE.T).max(axis=1)                    # best-matching fragment per candidate
        topk=[cand[i] for i in np.argsort(-sims)[:TOP_SIM]]
    else:
        topk=[]
    # rerank the chain strings the same way for the prompt
    if chain_strs:
        ss=(emb(chain_strs)@FE.T).max(axis=1)
        chain_strs=[chain_strs[i] for i in np.argsort(-ss)[:8]]
    # 5. answer
    opts_block="\n".join(f"  {k}. {v}" for k,v in sorted(opts.items()))
    ev = "\n".join(f"- {s}" for s in chain_strs) or "\n".join(f"- {nm}" for _,nm in topk) or "(none)"
    prompt_full=ANSWER.format(q=q, opts=opts_block, chains=ev)
    raw=call_llm(prompt_full, model=os.environ.get("HYKGE_LLM","gpt-5.4-mini"))
    pred=extract_letter(raw)
    return {"uid":uid,"gold":gold,"predicted":pred,"is_correct":pred==gold,
            "n_anchors":len(anchors),"n_chain_nodes":len(chain_nodes),
            "cand_top10":[nm for _,nm in topk[:10]],
            # --- full persistence for later analysis ---
            "hypothesis_output":ho_text,
            "ho_entities":ho_ents,
            "anchors":[n for _,n in anchors],
            # HyKGE injects reasoning CHAINS, not entities (paper Fig.3) — store what we actually send
            "kg_block":"\n".join(f"  {i+1}. {s}" for i,s in enumerate(chain_strs)),
            "top_entities":[nm for _,nm in topk[:30]],
            "chains_full":chain_strs,
            "prompt_full":prompt_full,
            "raw_response":raw}

def main():
    uids=list(B.values()); N=int(os.environ.get("HYKGE_N","0"))
    if N: uids=uids[:N]
    NW=int(os.environ.get("HYKGE_WORKERS","4"))
    save=CACHE/f"hykge_baseline_{os.environ.get('KG_OUT_TAG','329')}__{os.environ.get('HYKGE_LLM','gpt-5.4-mini').replace('/','_')}.json"
    out={"results":[]}; done_uids=set()
    if not N and save.exists():
        try:
            prev=json.load(open(save)).get("results",[])
            if len(prev)>10:
                out["results"]=prev; done_uids={r["uid"] for r in prev}
                print(f"resume: {len(done_uids)} already done", flush=True)
        except Exception: pass
    todo=[it for it in uids if it["uid"] not in done_uids]
    t0=time.time(); done=0
    def flush():
        nok=sum(1 for r in out["results"] if r.get("is_correct"))
        out["metadata"]={"n":len(out["results"]),"acc":100*nok/max(1,len(out["results"])),"max_hop":MAX_HOP,"top_sim":TOP_SIM}
        json.dump(out, open(save,"w"), indent=1)
    with ProcessPoolExecutor(max_workers=NW) as ex:
        futs={ex.submit(process_one,it):it["uid"] for it in todo}
        for fut in as_completed(futs):
            try: out["results"].append(fut.result())
            except Exception as e: print(f"  ERR {futs[fut]}: {e}", flush=True); continue
            done+=1
            if done%10==0 or done<=3:
                nok=sum(1 for r in out["results"] if r.get("is_correct"))
                flush()
                print(f"[{len(out['results'])}/{len(uids)}] (+{done}) acc={nok}/{len(out['results'])}={100*nok/len(out['results']):.1f}% elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    flush()
    print(f"saved -> {save}  acc={out['metadata']['acc']:.1f}%  n={out['metadata']['n']}", flush=True)

if __name__=="__main__":
    main()
