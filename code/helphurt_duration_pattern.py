"""#1 (trimmed): let the 'when does duration help' pattern emerge bottom-up from
help/hurt cases of the bc-on vs bc-off walker pair (isolates the duration effect).

HELP = bc flips wrong->right ; HURT = bc flips right->wrong.
For each case pull light features: temporal cue in Q, patient duration regime,
whether bc actually surfaced a candidate near the top. Then eyeball the split.
"""
import json, re
from pathlib import Path
CACHE = Path(__file__).resolve().parent.parent / "cache"

BC = {x['uid']: x for x in json.load(open(CACHE/"bench340_walker_fixABCmtS__gpt-5.4-mini.json"))['results']}
B0 = {x['uid']: x for x in json.load(open(CACHE/"bench340_walker_BC0__gpt-5.4-mini.json"))['results']}
Q  = {x['uid']: x for x in json.load(open(CACHE/"benchmark_bench340_v9.json"))}

common = set(BC)&set(B0)
HELP = [u for u in common if not B0[u].get('is_correct') and BC[u].get('is_correct')]
HURT = [u for u in common if B0[u].get('is_correct') and not BC[u].get('is_correct')]

CUE = re.compile(r'\b(\d+[\s-]*(hour|day|week|month|year)s?|hours?|days?|weeks?|months?|years?|'
                 r'chronic|acute|subacute|sudden|gradual|progressive|recurrent|intermittent|'
                 r'since|for the (past|last)|history of|ago|onset|duration)\b', re.I)

# distinct DURATION expressions, excluding patient age ("47-year-old", "a 45-year-old man")
AGE = re.compile(r'\b\d+[\s-]*year[\s-]*old\b', re.I)
DUR = re.compile(r'\b(\d+)[\s-]*(hour|day|week|month|year)s?\b', re.I)
def n_timelines(q):
    q2 = AGE.sub(' ', q)                      # strip patient age
    vals = set()
    for m in DUR.finditer(q2):
        vals.add((m.group(1), m.group(2).lower()))   # distinct (number, unit) pairs
    return len(vals), sorted(vals)

def regime(days):
    if days is None: return 'none'
    if days < 1:   return 'hyperacute(<1d)'
    if days < 14:  return 'acute(<2w)'
    if days < 42:  return 'subacute(2-6w)'
    if days < 365: return 'chronic(6w-1y)'
    return 'long(>1y)'

def cue_snippet(q):
    m = CUE.findall(q)
    hits = sorted({(h[0] if isinstance(h,tuple) else h).lower() for h in [x[0] for x in re.findall(r'('+CUE.pattern+r')', q, re.I)]})
    return hits[:6]

def row(u, tag):
    q = Q[u]['question']
    cues = sorted({g.group(0).lower() for g in CUE.finditer(q)})
    pdys = BC[u].get('patient_days')
    return {
        'uid': u, 'tag': tag, 'patient_days': pdys, 'regime': regime(pdys),
        'has_cue': bool(cues), 'n_cues': len(cues), 'cues': cues[:8],
        'gold': BC[u]['gold'], 'bc_pred': BC[u].get('predicted'), 'b0_pred': B0[u].get('predicted'),
        'n_cand': BC[u].get('n_walker_candidates'),
    }

print(f"HELP={len(HELP)}  HURT={len(HURT)}\n")
for tag,lst in [('HELP',HELP),('HURT',HURT)]:
    print(f"===== {tag} =====")
    for u in lst:
        r=row(u,tag); nt,tls=n_timelines(Q[u]['question'])
        print(f"  {u:14s} reg={r['regime']:16s} timelines={nt} {tls if nt>1 else ''} "
              f"gold={r['gold']} b0={r['b0_pred']}->bc={r['bc_pred']}")
    print()

def multi(lst):  # fraction with >=2 competing timelines
    cnts=[n_timelines(Q[u]['question'])[0] for u in lst]
    return sum(1 for c in cnts if c>=2), len(lst), sum(cnts)/len(lst)
hm,hn,ha=multi(HELP); um,un,ua=multi(HURT)
print("=== PATTERN SUMMARY: competing (non-age) timelines ===")
print(f"HELP: {hm}/{hn} have >=2 timelines   mean timelines/Q = {ha:.2f}")
print(f"HURT: {um}/{un} have >=2 timelines   mean timelines/Q = {ua:.2f}")
json.dump({'help':[row(u,'HELP') for u in HELP],'hurt':[row(u,'HURT') for u in HURT]},
          open(CACHE/"helphurt_duration_pattern.json","w"), indent=2, ensure_ascii=False)
print("\nsaved -> cache/helphurt_duration_pattern.json")
