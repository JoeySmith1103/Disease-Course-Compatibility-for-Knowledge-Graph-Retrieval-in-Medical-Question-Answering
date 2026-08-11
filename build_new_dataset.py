#!/usr/bin/env python3
"""Build a candidate benchmark from an external source into the pipeline's dataset format, so the
same verifier (and later the same retrieval) can run on it unchanged.

This step is deliberately only a CANDIDATE filter — it keeps questions that are clinical vignettes
containing a temporal expression. Whether a question is actually duration-critical is decided by
verify_duration_critical.py, which tests it; the keyword screen here only decides what is worth
paying to test, and its output must never be reported as "duration-critical".

  medbullets  308 USMLE Step 2/3 vignettes (ChallengeClinicalQA), 5 options — closest match to
              MedQA in style and length, and not part of the MedQA training distribution
  medmcqa     the local MedMCQA temporal split; only test.jsonl carries options and labels, and
              many items are factoid recall rather than vignettes, so a vignette screen applies
  mmlu        MMLU professional_medicine test split (272 items) — USMLE-style vignettes, small but
              the most widely reported medical benchmark in current papers

Not included: JAMA Clinical Challenge — ChallengeClinicalQA ships only links and a scraper, and
the articles sit behind a subscription, so the data cannot be redistributed or fetched here.
PubMedQA / BioASQ — abstract-level yes/no/maybe, no patient vignette and no duration to perturb.

Usage:  python3 pipeline/build_new_dataset.py medbullets
        python3 pipeline/build_new_dataset.py medmcqa
        python3 pipeline/build_new_dataset.py mmlu
"""
import csv, json, re, sys, urllib.request
from pathlib import Path

PIPE = Path(__file__).resolve().parent
MEDQA = Path("/Users/zengyuhua/Desktop/Master_thesis/Temporal-KG-RAG/datasets")
MB_URL = ("https://raw.githubusercontent.com/HanjieChen/ChallengeClinicalQA/main/"
          "medbullets/medbullets_op5.csv")

# a temporal expression of the presenting problem — not a proof of criticality, just a screen
TEMPORAL = re.compile(r"\b\d+\s*[- ]?(hour|day|week|month|year|minute)s?\b|"
                      r"\b(since|ago|for the (past|last)|history of|onset|duration|lasting)\b", re.I)
# a patient vignette rather than a bare factoid ("Most common site of osteosarcoma is:")
VIGNETTE = re.compile(r"\b(\d+[- ]?(year|month|day|week)[- ]?old|patient|presents|presented|"
                      r"\bman\b|\bwoman\b|\bboy\b|\bgirl\b|complains)", re.I)
LETTERS = "ABCDEFGHIJ"


def screen(q, min_chars):
    """Keyword screen — applied ONLY to MedMCQA.

    For MedBullets (308) and MMLU professional_medicine (272) the whole dataset is small enough to
    review by hand, so screening buys nothing and costs coverage: on MedBullets it discarded a
    study-design item about attrition over a 5-year follow-up and a milestone item whose age is
    conveyed entirely by the milestones themselves — both genuinely time-dependent, both invisible
    to a regex looking for "for 3 days". Screening those datasets would bake the script's
    single-axis blind spot into the corpus. MedMCQA is different: its pool is 182k questions with
    a median length of 65 characters, so some screen is unavoidable."""
    return len(q) >= min_chars and bool(TEMPORAL.search(q)) and bool(VIGNETTE.search(q))


def build_medbullets():
    raw = PIPE / "datasets/_raw_medbullets_op5.csv"
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MB_URL, raw)
    rows = list(csv.DictReader(open(raw)))
    out, opt_keys = [], ["opa", "opb", "opc", "opd", "ope"]
    for i, r in enumerate(rows):
        q = (r["question"] or "").strip()
        opts = {LETTERS[j]: (r.get(k) or "").strip() for j, k in enumerate(opt_keys)
                if (r.get(k) or "").strip()}
        # answer_idx is already a letter in this file; normalise defensively
        ans = (r.get("answer_idx") or "").strip().upper()
        if ans.isdigit(): ans = LETTERS[int(ans) - 1]
        if ans not in opts: continue          # no keyword screen: the whole set is reviewed
        out.append({"uid": f"mb_{i:04d}", "question": q, "options": opts, "answer": ans,
                    "source": "medbullets_op5", "link": r.get("link")})
    return out, len(rows)


def _hf_rows(dataset, config, split, cache_name):
    """Pull a whole split through the HF datasets-server (paged at 100)."""
    cache = PIPE / f"datasets/_raw_{cache_name}.json"
    if cache.exists():
        return json.load(open(cache))
    rows, off = [], 0
    while True:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
               f"&config={config}&split={split}&offset={off}&length=100")
        page = json.load(urllib.request.urlopen(url, timeout=90))
        rows += [r["row"] for r in page["rows"]]
        off += 100
        if off >= page["num_rows_total"]: break
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(cache, "w"))
    return rows


def build_medmcqa():
    """Original MedMCQA validation split — NOT the local MedMCQA_temporal_critical_split.

    That local file is a dead end for this purpose: all 1168 rows already carry
    is_temporal_critical=True from an earlier LLM filter, and the MedQA-329 cross-tab showed such
    judgement labels have no discriminative power (truly_tc 6.6% vs borderline_tc 6.5%). Building
    on it would inherit an unvalidated selection and inflate any rate measured afterwards. Its
    idx values run to 182738, so it was drawn from the 182k-row TRAIN split as well.

    Validation is used because it is the labelled split papers evaluate on (the official test
    split ships without answers). Note MedMCQA is mostly one-line factoid recall — median question
    length is 65 characters — so the vignette + length screen discards most of it by design."""
    rows = _hf_rows("openlifescienceai%2Fmedmcqa", "default", "validation", "medmcqa_validation")
    out = []
    for i, r in enumerate(rows):
        q = (r.get("question") or "").strip()
        opts = {LETTERS[j]: (r.get(k) or "").strip()
                for j, k in enumerate(["opa", "opb", "opc", "opd"]) if (r.get(k) or "").strip()}
        cop = r.get("cop")                       # 0-indexed in the HF release
        if not isinstance(cop, int) or not 0 <= cop < len(opts): continue
        if not screen(q, 200): continue
        out.append({"uid": f"mmc_{i:05d}", "question": q, "options": opts,
                    "answer": LETTERS[cop], "source": "medmcqa_validation",
                    "subject": r.get("subject_name")})
    return out, len(rows)


def build_mmlu():
    """MMLU professional_medicine via the HF datasets-server (no `datasets` install needed).
    The endpoint caps `length` at 100, so the split is pulled in pages."""
    cache = PIPE / "datasets/_raw_mmlu_professional_medicine.json"
    if cache.exists():
        rows = json.load(open(cache))
    else:
        rows, off = [], 0
        while True:
            url = ("https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu"
                   f"&config=professional_medicine&split=test&offset={off}&length=100")
            page = json.load(urllib.request.urlopen(url, timeout=60))
            rows += [r["row"] for r in page["rows"]]
            off += 100
            if off >= page["num_rows_total"]: break
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(cache, "w"))
    out = []
    for i, r in enumerate(rows):
        q = (r.get("question") or "").strip()
        ch = r.get("choices") or []
        opts = {LETTERS[j]: str(c).strip() for j, c in enumerate(ch) if str(c).strip()}
        ans = r.get("answer")           # 0-indexed int
        if not isinstance(ans, int) or not 0 <= ans < len(opts): continue
        # no keyword screen: 272 questions is small enough to review in full
        out.append({"uid": f"mmlu_{i:04d}", "question": q, "options": opts,
                    "answer": LETTERS[ans], "source": "mmlu_professional_medicine"})
    return out, len(rows)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "medbullets"
    items, total = {"medbullets": build_medbullets, "medmcqa": build_medmcqa,
                    "mmlu": build_mmlu}[name]()
    d = PIPE / f"datasets/{name}"; d.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(d / "benchmark.json", "w"), indent=1)
    ql = [len(x["question"]) for x in items]
    print(f"[{name}] {len(items)} / {total} kept  "
          f"(avg {sum(ql)/len(ql):.0f} chars, {sum(len(x['options']) for x in items)/len(items):.2f} options)")
    print(f"  -> {d/'benchmark.json'}")
    print("  NOTE: this is a CANDIDATE pool. Run verify_duration_critical.py to find out how many "
          "are actually duration-critical.")
