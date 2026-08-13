#!/usr/bin/env python3
"""Per-question 0–1 score for how much the diagnosis hinges on the patient's duration.

Feeds the adaptive-weight utility: w_temporal = criticality, w_semantic = 1 − criticality, so a
question whose answer turns on the time course leans on bc, and one where the duration is incidental
leans on the symptom match. A single global λ cannot express that difference; this can.

The vignette is the ONLY input. The options are deliberately withheld — a weight derived from the
answer choices would be reading the answer key into the retriever, which in an MCQ setting is
leakage no matter how indirect. This mirrors extract_patient_duration_llm.py, which also scores
from the vignette alone.

Scores go to datasets/<ds>/criticality.json as {uid: {score, rationale, decisive_axis}}, and are
appended incrementally so an interrupted run resumes instead of paying twice.

Usage:  DATASET=329 python3 pipeline/extract_duration_criticality.py
        DATASET=medbullets WORKERS=8 python3 pipeline/extract_duration_criticality.py
"""
import json, os, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "code"))
sys.path.insert(0, str(PIPE / "code" / "dkr_policy"))
from llm_client import call_llm

DS      = os.environ.get("DATASET", "329")
MODEL   = os.environ.get("CRIT_LLM", "gpt-5.4-mini")
WORKERS = int(os.environ.get("WORKERS", "8"))

PROMPT = """You are a clinical reasoning expert. Read the vignette below and judge ONE thing: how
much does identifying the correct diagnosis depend on the TIME COURSE of the patient's illness
(how long symptoms have lasted, how fast they came on, how they progressed)?

Score from 0.0 to 1.0:
  1.0  The time course is decisive. Change the duration and the leading diagnosis changes.
       (e.g. "3 days of joint pain" vs "3 years" — septic arthritis vs osteoarthritis)
  0.5  The time course narrows the differential but other findings carry equal weight.
  0.0  The time course is incidental. The diagnosis rests on findings, labs, or imaging that
       would read the same at any duration.

Judge the vignette as written. Do not speculate about what a different vignette would say, and do
not reward a case merely for MENTIONING a duration — a stated duration that does not discriminate
between plausible diagnoses scores low.

Vignette:
\"\"\"{q}\"\"\"

Reply with JSON only:
{{"score": <0.0-1.0>, "decisive_axis": "<symptom_duration|onset_speed|progression|latency|none>",
 "rationale": "<one sentence, max 25 words>"}}"""


def parse(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except Exception:
        return None
    s = d.get("score")
    if not isinstance(s, (int, float)) or not 0.0 <= s <= 1.0:
        return None
    return {"score": float(s), "decisive_axis": d.get("decisive_axis") or "none",
            "rationale": (d.get("rationale") or "")[:200]}


bench = json.load(open(PIPE / f"datasets/{DS}/benchmark.json"))
out_path = PIPE / f"datasets/{DS}/criticality.json"
done = json.load(open(out_path)) if out_path.exists() else {}
todo = [b for b in bench if b["uid"] not in done]
print(f"[{DS}] {len(bench)} 題，已完成 {len(done)}，待跑 {len(todo)}")


def one(b):
    for _ in range(3):
        r = parse(call_llm(PROMPT.format(q=b["question"]), model=MODEL))
        if r:
            return b["uid"], r
    return b["uid"], None


if todo:
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (uid, r) in enumerate(ex.map(one, todo), 1):
            if r:
                done[uid] = r
            if i % 25 == 0 or i == len(todo):
                json.dump(done, open(out_path, "w"), indent=1, ensure_ascii=False)
                print(f"  {i}/{len(todo)}", flush=True)
    json.dump(done, open(out_path, "w"), indent=1, ensure_ascii=False)

sc = [v["score"] for v in done.values()]
if sc:
    sc_sorted = sorted(sc)
    qq = lambda p: sc_sorted[min(int(p * len(sc_sorted)), len(sc_sorted) - 1)]
    from collections import Counter
    print(f"\n[{DS}] {len(sc)} 題 criticality")
    print(f"  p10={qq(.10):.2f} p25={qq(.25):.2f} p50={qq(.50):.2f} p75={qq(.75):.2f} p90={qq(.90):.2f}")
    print(f"  分佈: {dict(sorted(Counter(round(s,1) for s in sc).items()))}")
    print(f"  decisive_axis: {dict(Counter(v['decisive_axis'] for v in done.values()).most_common())}")
    print(f"  -> {out_path}")
