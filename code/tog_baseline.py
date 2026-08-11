#!/usr/bin/env python3
"""Faithful Think-on-Graph (ToG) baseline on our UMLS/SNOMED Neo4j KG.

Mirrors GasolSun36/ToG: an LLM-guided beam search over the graph.
Per depth d (<= MAX_DEPTH), for each frontier (topic) entity:
  1. relation_search : collect distinct RELA relations on the entity's edges (Neo4j)
  2. relation_prune  : LLM picks top-WIDTH relations relevant to the question, with scores
                       (ToG's signature LLM relation pruning; prompt mirrors prompt_list.py)
  3. entity_search   : fetch entities reached via each pruned relation
  4. entity_score    : LLM scores the candidate tail entities 0-1 (sum 1) for the question,
                       times the relation score  <-- ToG's signature LLM entity pruning
  5. entity_prune    : keep top-WIDTH candidates -> next frontier; append (e, rel, e') chains
  6. reasoning       : LLM sufficiency check; if Yes -> generate answer and stop
If depth exhausted, answer from accumulated chains (half_stop).
WIDTH=3, MAX_DEPTH=3 (ToG defaults). Records the explored candidate entities (for recall@10)
and the final answer (for accuracy).

Fidelity notes vs the paper (arXiv 2307.07697) / GasolSun36-IDEA-FinAI ToG:
  * Relation AND entity pruning are both LLM-driven, as in the flagship ToG. (ToG-R, the
    efficiency variant, replaces entity pruning with RANDOM sampling — not embeddings.)
    TOG_ENTITY_PRUNE=cos restores the earlier SapBERT-cosine scoring for ablation only.
  * TOG_REL_THRESH=0.2 mirrors the paper's relation-score cut-off before taking top-W.
  * Candidate tails shown to the LLM are capped at TOG_ENT_CAP=20 (ToG likewise subsamples
    when a relation has very many tails); unshown tails score 0 and drop out of the beam.
  * Topic entities come from our shared LLM-DDx seed set rather than question NER, so every
    retrieval method under comparison starts from the SAME seeds.
"""
import json, os, random, re, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; CACHE = ROOT/"cache"
# ROOT is pipeline/; the modules live in pipeline/code and pipeline/code/dkr_policy
sys.path.insert(0, str(ROOT/"code")); sys.path.insert(0, str(ROOT/"code"/"dkr_policy"))
os.environ.setdefault("DKR_MATRIX_PKL", "cache/umls_broad_embeddings_sapbert.pkl")

WIDTH = int(os.environ.get("TOG_WIDTH", "3"))
MAX_DEPTH = int(os.environ.get("TOG_DEPTH", "3"))
REL_LIMIT = 60        # cap relations shown to the LLM per entity
ENT_LIMIT = 200       # cap entities fetched per (entity, relation)
ENT_CAP = int(os.environ.get("TOG_ENT_CAP", "20"))          # tails shown to the LLM per relation
REL_THRESH = float(os.environ.get("TOG_REL_THRESH", "0.2"))  # paper's relation-score cut-off
ENTITY_PRUNE = os.environ.get("TOG_ENTITY_PRUNE", "llm").lower()  # 'llm' (faithful) | 'cos'

B = {x["uid"]: x for x in json.load(open(os.environ.get("BENCH_PATH", str(CACHE/"benchmark_bench329_clean.json"))))}
seeds_c = json.load(open(os.environ.get("SEEDS_PATH", str(CACHE/"bench340_seeds_multitype.json"))))
sym_c   = json.load(open(os.environ.get("SYM_PATH", str(CACHE/"bench340_symptoms.json"))))

REL_PRUNE = """You are exploring a medical knowledge graph to answer a question.
Question: {q}
Topic entity: {ent}
Available relations from this entity:
{rels}
Retrieve up to {w} relations most useful for answering the question and rate each
0-1. Output exactly, one per line: relation (score). No other text."""

ENT_PRUNE = """Please score the entities' contribution to the question on a scale from 0 to 1
(the sum of the scores of all entities is 1).
Question: {q}
Relation: {rel}
Entities:
{ents}
Output exactly one line per entity, in the form: entity (score). No other text."""

SUFF = """Question: {q}
Retrieved knowledge-graph triples:
{chains}
Given these triples and your own knowledge, is this sufficient to answer the question?
Answer ONLY Yes or No."""

ANSWER = """You are an expert medical diagnostician. Answer the multiple-choice question.
Question:
{q}
Options:
{opts}
Knowledge-graph evidence (entity - relation - entity), retrieved by Think-on-Graph:
{chains}
Reason briefly (<150 words), then give the final answer as a single letter in <a></a> tags."""

def extract_letter(raw):
    if not raw: return None
    for pat in [r"<a>\s*([A-J])\s*</a>", r"\\boxed\{\s*([A-J])\s*\}",
                r"\banswer\s*(?:is|:)\s*\**\(?([A-J])\)?\b"]:
        m=re.search(pat, raw, re.IGNORECASE)
        if m: return m.group(1).upper()
    return None

_G = {}
def _load():
    if _G: return
    from duration_kg_rag import _load_umls_emb
    idx,enc=_load_umls_emb()           # single SapBERT matrix load (DKR_MATRIX_PKL)
    mat=idx["matrix"]; mat=mat/np.maximum(np.linalg.norm(mat,axis=1,keepdims=True),1e-12)
    _G["mat"]=mat; _G["cuis"]=idx["cuis"]; _G["cui2idx"]={c:i for i,c in enumerate(idx["cuis"])}; _G["enc"]=enc
    from umls_neo4j import get_driver; _G["drv"]=get_driver()

def qemb(text):
    if isinstance(text,(list,tuple)): text=" ".join(map(str,text))
    text=str(text or "").strip() or "medical case"
    v=_G["enc"].encode([text]); return (v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12))[0]
def cos_to_q(cui, qv):
    j=_G["cui2idx"].get(cui,-1)
    return float(_G["mat"][j]@qv) if j>=0 else 0.0

def resolve_seeds(uid):
    names=(seeds_c.get(uid) or {}).get("seeds",[])[:12]
    if not names: return []
    e=_G["enc"].encode(names); e=e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
    out=[]
    for i,nm in enumerate(names):
        sc=_G["mat"]@e[i]; j=int(np.argmax(sc))
        if sc[j]>0.55: out.append((list(_G["cui2idx"].keys())[j] if False else None, nm))
    # resolve to cui + name via neo4j (top match cui)
    res=[]
    cuis=list(_G["cui2idx"].keys())
    for i,nm in enumerate(names):
        sc=_G["mat"]@e[i]; j=int(np.argmax(sc))
        if sc[j]>0.55: res.append((cuis[j], nm))
    # dedup by cui
    seen=set(); ded=[]
    for c,n in res:
        if c in seen: continue
        seen.add(c); ded.append((c,n))
    return ded[:WIDTH*2]

def relations_of(cui):
    with _G["drv"].session() as s:
        rows=s.run("MATCH (a:Concept {CUI:$c})-[r]-(b:Concept) "
                   "WHERE r.RELA IS NOT NULL RETURN DISTINCT r.RELA AS rela LIMIT $lim",
                   c=cui, lim=REL_LIMIT).data()
    return [r["rela"] for r in rows]

def entities_via(cui, rela):
    with _G["drv"].session() as s:
        rows=s.run("MATCH (a:Concept {CUI:$c})-[r {RELA:$rela}]-(b:Concept) "
                   "RETURN DISTINCT b.CUI AS cui, b.name AS name LIMIT $lim",
                   c=cui, rela=rela, lim=ENT_LIMIT).data()
    return [(r["cui"], r["name"]) for r in rows if r["cui"]]

def parse_rel_scores(raw, valid):
    out=[]
    for ln in (raw or "").splitlines():
        m=re.match(r"\s*(.+?)\s*\(?\s*([01](?:\.\d+)?)\s*\)?\s*$", ln.strip())
        if m:
            rel=m.group(1).strip().strip("-* ").lower()
            if rel in valid:
                try: out.append((rel, float(m.group(2))))
                except: pass
    return out

def parse_ent_scores(raw, names):
    """Map the LLM's `entity (score)` lines back onto the candidate names (case-insensitive,
    tolerating minor restatement). Returns {name: score}."""
    lookup = {n.strip().lower(): n for n in names}
    out = {}
    for ln in (raw or "").splitlines():
        m = re.match(r"\s*(.+?)\s*\(\s*([01](?:\.\d+)?)\s*\)\s*$", ln.strip())
        if not m:
            continue
        key = m.group(1).strip().strip("-*0123456789. ").lower()
        nm = lookup.get(key)
        if nm is None:  # fall back to a containment match
            nm = next((orig for k, orig in lookup.items() if k and (k in key or key in k)), None)
        if nm is not None:
            try:
                out[nm] = float(m.group(2))
            except ValueError:
                pass
    return out


def entity_scores(q, rela, cands, qv, call_llm):
    """ToG entity prune. Flagship ToG uses the LLM to score candidate tail entities; we mirror
    that. Returns a score per candidate (aligned with `cands`).

    TOG_ENTITY_PRUNE=cos restores the previous SapBERT-cosine scoring (ablation only).
    """
    if ENTITY_PRUNE == "cos":
        return [cos_to_q(c, qv) for c, _ in cands]
    # ToG subsamples when a relation has very many tails; unshown tails score 0 (dropped).
    shown = cands
    if len(shown) > ENT_CAP:
        shown = random.Random(hash((q, rela)) & 0xffffffff).sample(cands, ENT_CAP)
    names = [n for _, n in shown]
    raw = call_llm(ENT_PRUNE.format(q=q, rel=rela, ents="\n".join(f"- {n}" for n in names)),
                   model=os.environ.get("TOG_LLM", "gpt-5.4-mini"))
    scored = parse_ent_scores(raw, names)
    if not scored:  # LLM unparseable → uniform over the shown subset (keeps the beam alive)
        scored = {n: 1.0 / max(1, len(names)) for n in names}
    return [scored.get(n, 0.0) for _, n in cands]


def process_one(item):
    from llm_client import call_llm
    _load()
    uid=item["uid"]; gold=item["answer"]; opts=item["options"]
    q=item["question"]
    qv=qemb((sym_c.get(uid) or {}).get("symptoms","") or q)
    seeds=resolve_seeds(uid)
    frontier=seeds[:WIDTH]; visited={c for c,_ in seeds}
    chains=[]; scored_entities={}  # cui -> (name, score)
    answered=None
    for depth in range(MAX_DEPTH):
        new_cands=[]  # (src_name, rela, cui, name, score)
        for cui,ename in frontier:
            rels=relations_of(cui)
            if not rels: continue
            raw=call_llm(REL_PRUNE.format(q=q, ent=ename, rels="\n".join(f"- {r}" for r in rels), w=WIDTH),
                         model=os.environ.get("TOG_LLM","gpt-5.4-mini"))
            scored_rels=parse_rel_scores(raw, set(rels))
            # paper: drop relations below the score threshold, then take top-W
            scored_rels=[(r,s) for r,s in scored_rels if s >= REL_THRESH]
            scored_rels.sort(key=lambda x:-x[1])
            pruned=scored_rels[:WIDTH]
            if not pruned:  # fallback: keep clinically-useful default relations
                pruned=[(r,0.5) for r in rels[:WIDTH]]
            for rela,rscore in pruned:
                cands=[(nc,nn) for nc,nn in entities_via(cui, rela) if nc not in visited]
                if not cands: continue
                # ToG entity prune: the LLM scores the candidate tails (not embeddings)
                escores=entity_scores(q, rela, cands, qv, call_llm)
                for (ncui,nname),escore in zip(cands, escores):
                    new_cands.append((ename,rela,ncui,nname, escore*max(rscore,0.1)))
        if not new_cands: break
        # entity prune: top-WIDTH by score (beam), but record all for recall
        for ename,rela,ncui,nname,es in new_cands:
            if ncui not in scored_entities or es>scored_entities[ncui][1]:
                scored_entities[ncui]=(nname,es)
        new_cands.sort(key=lambda x:-x[4])
        beam=new_cands[:WIDTH]
        frontier=[(c,n) for _,_,c,n,_ in beam]
        for c,_ in frontier: visited.add(c)
        chains += [f"{sn} -[{rel}]- {nn}" for sn,rel,_,nn,_ in beam]
        # reasoning: sufficient?
        suff=call_llm(SUFF.format(q=q, chains="\n".join(chains[-20:])),
                      model=os.environ.get("TOG_LLM","gpt-5.4-mini"))
        if suff and suff.strip().lower().startswith("y"):
            break
    opts_block="\n".join(f"  {k}. {v}" for k,v in sorted(opts.items()))
    prompt_full=ANSWER.format(q=q, opts=opts_block, chains="\n".join(chains) or "(none)")
    raw=call_llm(prompt_full, model=os.environ.get("TOG_LLM","gpt-5.4-mini"))
    pred=extract_letter(raw)
    ranked=sorted(scored_entities.items(), key=lambda kv:-kv[1][1])
    return {"uid":uid,"gold":gold,"predicted":pred,"is_correct":pred==gold,
            "n_explored":len(scored_entities),
            "cand_top10":[nm for _,(nm,_) in ranked[:10]],
            # --- full persistence for later analysis ---
            # kg_block MUST be the evidence that actually went into prompt_full, i.e. the chains.
            # It used to hold the scored entity ranking instead, which is a different
            # representation of the same walk: harmless while the frozen prompt was replayed
            # verbatim, but silently wrong once prompts are re-rendered FROM kg_block — that
            # swapped ToG's reasoning chains for a bare entity list under ToG's name.
            "kg_block":"\n".join(f"  {i+1}. {c}" for i,c in enumerate(chains)) or "",
            "entity_ranking":"\n".join(f"  {i+1}. {nm} (score={sc:.3f})" for i,(_,(nm,sc)) in enumerate(ranked[:30])),
            "chains_full":chains,
            "seeds":[n for _,n in seeds],
            "prompt_full":prompt_full,
            "raw_response":raw}

def main():
    uids=list(B.values())
    N=int(os.environ.get("TOG_N","0"))
    if N: uids=uids[:N]
    NW=int(os.environ.get("TOG_WORKERS","4"))
    save=CACHE/f"tog_baseline_{os.environ.get('KG_OUT_TAG','329')}__{os.environ.get('TOG_LLM','gpt-5.4-mini').replace('/','_')}.json"
    # resume: keep prior results, skip done uids (unless smoke N given)
    out={"results":[]}; done_uids=set()
    if not N and save.exists():
        try:
            prev=json.load(open(save)).get("results",[])
            if len(prev)>10:  # ignore tiny smoke files
                out["results"]=prev; done_uids={r["uid"] for r in prev}
                print(f"resume: {len(done_uids)} already done", flush=True)
        except Exception: pass
    todo=[it for it in uids if it["uid"] not in done_uids]
    t0=time.time(); done=0
    def flush():
        nok=sum(1 for r in out["results"] if r.get("is_correct"))
        out["metadata"]={"n":len(out["results"]),"acc":100*nok/max(1,len(out["results"])),"width":WIDTH,"depth":MAX_DEPTH}
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
