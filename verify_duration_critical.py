#!/usr/bin/env python3
"""Verify duration-criticality by EXPERIMENT, not by asking an LLM to label it.

A question is duration-critical iff changing ONLY the stated duration changes the correct
answer. Previous filters asked a model "is this duration-critical?" — a judgement that, on a
100-question spot check, disagreed with deeper counterfactual analysis ~60% of the time. This
script tests the property instead of labelling it:

  A. CONTROL   answer the ORIGINAL question, N samples. If the model cannot get the original
               right, a flip on the perturbed version proves nothing → verdict "control_fail".
  B. PERTURB   rewrite ONLY the duration (acute↔chronic), everything else byte-identical.
               A separate check confirms the rest of the vignette was not altered.
  C. TEST      answer the PERTURBED question, N samples, WITHOUT showing the original answer
               (no anchoring — this is why B and C are separate calls).

Both sides use a supermajority (≥80% / ≤20% of samples on gold), not a bare majority — otherwise
resampling noise at temperature 1 masquerades as a duration effect.

  verdict = duration_critical      original settled on gold, perturbed almost never picks gold
          = not_duration_critical  original settled on gold, perturbed still settles on gold
          = ambiguous              perturbed answer sits between the thresholds — unstable, and
                                   deliberately not forced into either bucket
          = control_fail           the model cannot answer the ORIGINAL reliably, so its answer to
                                   the perturbed version is not trustworthy evidence either way
          = perturb_fail           duration could not be changed (none stated, or the rewrite
                                   altered more than the duration)

Every call is stored, so any verdict can be audited afterwards.

Usage:
  DATASET=329 MODEL=gpt-5.4-mini N_SAMPLES=3 WORKERS=8 python3 verify_duration_critical.py
Output: pipeline/verification/<ds>_duration_critical_<model>.json  (resumable)
"""
import json, os, re, sys, importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "code"))
from llm_client import call_llm

DATASET   = os.environ.get("DATASET", "329")
MODEL     = os.environ.get("MODEL", "gpt-5.4-mini")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "3"))
WORKERS   = int(os.environ.get("WORKERS", "8"))
LIMIT     = int(os.environ.get("LIMIT", "0"))          # 0 = all
OUT = PIPE / "verification"; OUT.mkdir(exist_ok=True)
OUT_FILE = OUT / f"{DATASET}_duration_critical_{MODEL.replace('/','_')}.json"

_spec = importlib.util.spec_from_file_location("P", PIPE / "prompts.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)

bench = json.load(open(PIPE / f"datasets/{DATASET}/benchmark.json"))
if LIMIT: bench = bench[:LIMIT]


def ob(o): return "\n".join(f"  {k}. {v}" for k, v in sorted(o.items()))


PERTURB = """You are a medical expert. Rewrite this clinical vignette changing ONLY the symptom duration.

Vignette:
{question}

Rules:
1. Identify the duration of the presenting problem (e.g. "for 3 days", "2-month history").
2. Replace it with a duration on the OPPOSITE time scale — acute (hours/days) becomes chronic
   (months/years), or chronic becomes acute. Make the change large and unambiguous.
3. Change NOTHING else. Every symptom, sign, lab value, age, sex and number must stay identical.
4. If the vignette states no duration for the presenting problem, set "has_duration" to false.

Return strict JSON only:
{{"has_duration": true/false,
  "original_duration": "the exact phrase you replaced",
  "new_duration": "the replacement phrase",
  "modified_question": "the full vignette with ONLY that phrase changed"}}"""

ANSWER = """You are an expert medical diagnostician. Answer the multiple-choice question.

Question:
{question}

Options:
{options_block}

Reason briefly, then give the final answer as a single letter in <a></a> tags."""


def extract(raw):
    for p in [r"<a>\s*([A-L])\s*</a>", r"\banswer\s*(?:is|:)\s*\**\(?([A-L])\)?\b",
              r"\\boxed\{\s*([A-L])\s*\}"]:
        m = re.search(p, raw or "", re.I)
        if m: return m.group(1).upper()
    return None


def vote(question, options, n):
    """Answer a question n times independently; return the raw letters."""
    return [extract(call_llm(ANSWER.format(question=question, options_block=ob(options)),
                             model=MODEL)) for _ in range(n)]


# A bare majority is too weak to attribute a flip to duration: at temperature 1 a model that
# answers gold 3/5 times will "flip" on resampling alone. Both sides are therefore held to a
# supermajority, and everything in between is reported as ambiguous rather than forced.
HI = -(-4 * N_SAMPLES // 5)      # ceil(0.8N) — how often gold must appear to count as settled
LO = (1 * N_SAMPLES) // 5        # floor(0.2N) — how rarely gold may appear to count as a flip


def diff_blocks(orig, mod):
    """Word-level diff between the two vignettes: list of (word_index, removed, added).

    Naive string replacement cannot judge this: "that started 2 hours ago" -> "for 2 years" is a
    legitimate duration-only edit that produces several diff hunks because it changes surrounding
    grammar. What separates a clean edit from a contaminated one is LOCALITY — a duration rewrite
    touches one neighbourhood of the vignette, while an edit that also altered a symptom or lab
    value leaves hunks scattered across it."""
    import difflib
    a = re.sub(r"\s+", " ", (orig or "")).strip().lower().split()
    b = re.sub(r"\s+", " ", (mod or "")).strip().lower().split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append((i1, " ".join(a[i1:i2]), " ".join(b[j1:j2])))
    return out


TIME_WORD = re.compile(r"\b(hour|day|week|month|year|minute|night|morning|acute|chronic|ago|"
                       r"since|history|onset|duration|lasting|past)", re.I)


def integrity(orig, mod):
    """(ok, blocks) — ok iff all edits sit in ONE local neighbourhood and mention a time unit."""
    blocks = diff_blocks(orig, mod)
    if not blocks:
        return False, blocks
    span = blocks[-1][0] - blocks[0][0] + len(blocks[-1][1].split())
    changed = sum(max(len(r.split()), len(a.split())) for _, r, a in blocks)
    text = " ".join(r + " " + a for _, r, a in blocks)
    # span: every hunk within a ~20-word window (one phrase, not scattered edits)
    # changed: the rewrite may not carry more than ~25 words of new content
    return bool(TIME_WORD.search(text)) and span <= 20 and changed <= 25, blocks


def verify(item):
    q, opts, gold = item["question"], item["options"], item["answer"]
    rec = {"uid": item["uid"], "gold": gold}

    # A. CONTROL — is the original answer settled for this model?
    ctrl_all = vote(q, opts, N_SAMPLES)
    rec["control_samples"] = ctrl_all
    rec["control_gold_hits"] = ctrl_all.count(gold)
    if rec["control_gold_hits"] < HI:
        rec["verdict"] = "control_fail"
        return rec

    # B. PERTURB — change only the duration
    raw = call_llm(PERTURB.format(question=q), model=MODEL)
    m = re.search(r"\{[\s\S]*\}", raw or "")
    try:
        pert = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        pert = {}
    rec["perturb"] = {k: pert.get(k) for k in
                      ("has_duration", "original_duration", "new_duration")}
    if not pert.get("has_duration") or not pert.get("modified_question"):
        rec["verdict"] = "perturb_fail"; rec["perturb_reason"] = "no duration stated"
        return rec
    rec["modified_question"] = pert["modified_question"]
    ok, blocks = integrity(q, pert["modified_question"])
    rec["only_duration_changed"], rec["diff_blocks"] = ok, blocks
    if not ok:
        rec["verdict"] = "perturb_fail"
        rec["perturb_reason"] = f"rewrite touched {len(blocks)} region(s), not a clean duration edit"
        return rec

    # C. TEST — answer the perturbed vignette fresh (gold never shown, so no anchoring)
    new_all = vote(pert["modified_question"], opts, N_SAMPLES)
    rec["perturbed_samples"] = new_all
    hits = new_all.count(gold)
    rec["perturbed_gold_hits"] = hits
    top = Counter([x for x in new_all if x]).most_common(1)
    rec["perturbed_answer"] = top[0][0] if top else None
    if hits <= LO:      rec["verdict"] = "duration_critical"
    elif hits >= HI:    rec["verdict"] = "not_duration_critical"
    else:               rec["verdict"] = "ambiguous"
    return rec


def main():
    done = {}
    if OUT_FILE.exists():
        try: done = {r["uid"]: r for r in json.load(open(OUT_FILE))["results"]}
        except Exception: done = {}
    todo = [b for b in bench if b["uid"] not in done]
    print(f"[verify] {DATASET} · {MODEL} · N={N_SAMPLES} · {len(todo)} to go "
          f"({len(done)} cached)", flush=True)
    results = list(done.values())

    def save():
        c = Counter(r["verdict"] for r in results)
        json.dump({"dataset": DATASET, "model": MODEL, "n_samples": N_SAMPLES,
                   "thresholds": {"control_gold_hits_min": HI, "perturbed_gold_hits_max": LO},
                   "n": len(results), "verdicts": dict(c), "results": results},
                  open(OUT_FILE, "w"), indent=1)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(verify, b): b["uid"] for b in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try: results.append(f.result())
            except Exception as e: print(f"  ERR {futs[f]}: {repr(e)[:90]}", flush=True); continue
            if i % 20 == 0:
                save(); c = Counter(r["verdict"] for r in results)
                print(f"  [{len(results)}/{len(bench)}] {dict(c)}", flush=True)
    save()
    c = Counter(r["verdict"] for r in results)
    tested = c["duration_critical"] + c["not_duration_critical"]
    print(f"\n[verify] {DATASET}: {dict(c)}")
    if tested:
        print(f"  真正 duration-critical: {c['duration_critical']}/{tested} = "
              f"{100*c['duration_critical']/tested:.1f}% （在控制組通過的題目中）")
    print(f"  -> {OUT_FILE}")


if __name__ == "__main__":
    main()
