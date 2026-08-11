"""#2 (v2): validate LLM per-disease duration intervals against StatPearls/NCBI,
comparing LIKE-FOR-LIKE on canonical temporal axes (acute / subacute / chronic) and
using INTERVAL OVERLAP as the primary agreement metric (interpretable as "is the LLM
interval consistent with the literature range?"). Axis-implausible extractions dropped.

Prior v1 mixed axes (acute-episode vs chronic-baseline for the same disease) via geo-mean,
which conflated different clinical durations → noisy ρ=0.68. v2 removes that confound.
"""
import json, math, statistics
from pathlib import Path
CACHE = Path(__file__).resolve().parent.parent / "cache"
NCBI = CACHE / "ncbi_phi_combined_mentions.jsonl"
LLM  = CACHE / "per_disease_durations_on_demand.jsonl"

# role string -> canonical axis (None = different temporal axis, excluded)
def axis(role):
    r = (role or "").lower()
    if "age_of_onset" in r or "incubation" in r or "latency" in r or "onset_to_presentation" in r \
       or "prognosis" in r or "survival" in r or "penetrance" in r or "time_to_resolution" in r:
        return None
    if "hyperacute" in r: return "acute"          # fold hyperacute into acute envelope
    if "acute" in r and "subacute" not in r: return "acute"
    if "subacute" in r: return "subacute"
    if "chronic" in r or "lifelong" in r: return "chronic"
    if "symptomatic_course" in r or "total_course" in r or "total_disease_course" in r: return "course"
    return None

# plausibility window per axis (days) — drop extraction errors outside these
PLAUS = {"acute": (0.02, 60), "subacute": (5, 120), "chronic": (14, 40*365), "course": (0.5, 40*365)}

def norm(s): return " ".join((s or "").lower().split())

def per_axis(path):
    """disease -> axis -> (lo, hi, mid) using geo-mean envelope of that axis' mentions."""
    out = {}
    for line in open(path):
        d = json.loads(line); dis = norm(d.get("disease"))
        buckets = {}
        for m in d.get("mentions", []):
            ax = axis(m.get("role")); lo, hi = m.get("min_days"), m.get("max_days")
            if not ax or lo is None or hi is None or lo <= 0 or hi <= 0: continue
            plo, phi = PLAUS[ax]
            mid = math.sqrt(lo*hi)
            if mid < plo or mid > phi: continue          # drop axis-implausible extraction
            buckets.setdefault(ax, []).append((lo, hi, mid))
        if buckets:
            out[dis] = {ax: (min(l for l,_,_ in v), max(h for _,h,_ in v),
                             math.exp(statistics.fmean(math.log(m) for _,_,m in v)))
                        for ax, v in buckets.items()}
    return out

gt, llm = per_axis(NCBI), per_axis(LLM)

def overlap(a, b):  # do [lo,hi] envelopes intersect?
    return a[0] <= b[1] and b[0] <= a[1]

print("=== LLM vs StatPearls, matched per canonical axis, interval-overlap ===\n")
rows = []
allpairs = []
for ax in ["acute", "subacute", "chronic", "course"]:
    pairs = [(d, gt[d][ax], llm[d][ax]) for d in (set(gt)&set(llm))
             if ax in gt[d] and ax in llm[d]]
    if not pairs: continue
    n = len(pairs)
    ov = 100*sum(1 for _,g,l in pairs if overlap(g, l))/n
    ratios = [l[2]/g[2] for _,g,l in pairs]
    logr = [math.log10(r) for r in ratios]
    med_ratio = 10**statistics.median(logr)
    w3 = 100*sum(1 for r in ratios if 1/3 <= r <= 3)/n
    print(f"  {ax:9s} n={n:4d}  interval-overlap={ov:5.1f}%  median LLM/SP ratio={med_ratio:4.2f}x  within-3x(mid)={w3:4.1f}%")
    rows.append({"axis": ax, "n": n, "overlap_pct": ov, "median_ratio": med_ratio, "within_3x_pct": w3})
    allpairs += [(d,ax,g,l) for d,g,l in pairs]

# pooled
n = len(allpairs)
ov = 100*sum(1 for _,_,g,l in allpairs if overlap(g,l))/n
ratios = [l[2]/g[2] for _,_,g,l in allpairs]
med = 10**statistics.median([math.log10(r) for r in ratios])
# ordinal corr of midpoints (pooled, log)
lx=[math.log10(g[2]) for _,_,g,l in allpairs]; ly=[math.log10(l[2]) for _,_,g,l in allpairs]
def rank(v):
    o=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
    for p,i in enumerate(o): r[i]=p
    return r
rx,ry=rank(lx),rank(ly); mrx,mry=statistics.fmean(rx),statistics.fmean(ry)
sp=sum((a-mrx)*(b-mry) for a,b in zip(rx,ry))/(math.sqrt(sum((a-mrx)**2 for a in rx))*math.sqrt(sum((b-mry)**2 for b in ry)))
print(f"\n  POOLED  n={n}  interval-overlap={ov:.1f}%  median ratio={med:.2f}x  Spearman(mid)={sp:.3f}")

json.dump({"per_axis": rows, "pooled": {"n": n, "overlap_pct": ov, "median_ratio": med, "spearman": sp}},
          open(CACHE/"validate_llm_durations_vs_statpearls_v2.json","w"), indent=2)
print("\nsaved -> cache/validate_llm_durations_vs_statpearls_v2.json")
