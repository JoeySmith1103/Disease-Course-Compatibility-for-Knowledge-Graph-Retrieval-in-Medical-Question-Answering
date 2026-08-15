#!/usr/bin/env python3
"""What kind of entity does each question ask for — a disease, an organism, a drug, a procedure?

Feeds a fourth judgement dimension. cos measures how much a concept reads like the vignette, and
the vignette describes symptoms, so concepts that are not symptom-like are systematically
under-ranked no matter how relevant they are. Measured on the pools: 639 organism candidates on 329
reach the top-10 twice, 211 on MedBullets reach it zero times, and 721 procedure candidates on
MedBullets yield 54 slots. A question asking "which organism" is therefore ranked by a signal that
structurally disfavours the answer.

The retired walker patched this with a flat +0.07 for organism and drug roles. This asks instead,
per question, what the answer is supposed to be, and lets the weight follow — a judgement rather
than a constant, and one that also covers procedure questions (27% of MedBullets), which the flat
bonus never did.

The stem and vignette are the only input. The options are withheld: reading them would let the
answer set decide which candidates get promoted, which is leakage however indirect. This mirrors
extract_duration_criticality.py.

Usage:  DATASET=329 python3 pipeline/extract_answer_type.py
"""
import json, os, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "code"))
sys.path.insert(0, str(PIPE / "code" / "dkr_policy"))
from llm_client import call_llm

DS      = os.environ.get("DATASET", "329")
MODEL   = os.environ.get("TYPE_LLM", "gpt-5.4-mini")
WORKERS = int(os.environ.get("WORKERS", "8"))
TYPES = {"disease", "organism", "drug", "procedure", "finding", "other"}

PROMPT = """Read the clinical vignette and its final question. Say what KIND of thing the correct
answer is — not what the answer is, only its category.

  disease    a diagnosis or condition            ("most likely diagnosis", "underlying condition")
  organism   a pathogen                          ("causal organism", "which bacterium")
  drug       a medication or drug class          ("most appropriate pharmacotherapy")
  procedure  a test, imaging study or intervention ("next best step", "most appropriate test")
  finding    a sign, symptom, or lab/exam finding ("which additional finding would you expect")
  other      anything else (mechanism, risk factor, prognosis, epidemiology)

Judge from the question as written. If the stem names the category explicitly, use it.

Vignette and question:
\"\"\"{q}\"\"\"

Reply with JSON only: {{"answer_type": "<one of the six>", "evidence": "<the words in the stem that
decide it, max 12 words>"}}"""


def parse(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except Exception:
        return None
    t = (d.get("answer_type") or "").strip().lower()
    return {"answer_type": t, "evidence": (d.get("evidence") or "")[:120]} if t in TYPES else None


bench = json.load(open(PIPE / f"datasets/{DS}/benchmark.json"))
out_path = PIPE / f"datasets/{DS}/answer_type.json"
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
            if i % 50 == 0 or i == len(todo):
                json.dump(done, open(out_path, "w"), indent=1, ensure_ascii=False)
                print(f"  {i}/{len(todo)}", flush=True)
    json.dump(done, open(out_path, "w"), indent=1, ensure_ascii=False)

from collections import Counter
c = Counter(v["answer_type"] for v in done.values())
print(f"\n[{DS}] {len(done)} 題 answer_type: {dict(c.most_common())}")
print(f"  -> {out_path}")
