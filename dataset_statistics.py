#!/usr/bin/env python3
"""Descriptive statistics for the benchmark splits, in the style of the ReinRAG dataset table
(question/option/answer length, vocabulary, answer balance) plus the duration-specific columns
this thesis needs (duration coverage, temporal role, seed/symptom counts).

Every dataset is reported twice: over ALL its questions, and over the DURATION-CRITICAL subset
only. Reporting only the aggregate hides the thing the thesis is about — whether the questions
whose answer actually depends on time look different from the rest (longer? more options? shifted
duration distribution?). The subset is taken from the hand review where one exists
(verification/manual_read_<ds>.jsonl) and otherwise from the perturbation script's verdicts, and
the source is printed so the two are never silently mixed.

Usage:  python3 pipeline/dataset_statistics.py [dataset ...]   # default: 329 1273
"""
import json, re, statistics, sys, glob
from pathlib import Path
from collections import Counter

PIPE = Path(__file__).resolve().parent
# candidate pools built by build_new_dataset.py have no retrieval inputs yet, so every lookup
# below has to tolerate a missing file rather than assume the 329/1273 layout
DATASETS = sys.argv[1:] or ["329", "1273"]
WORD = re.compile(r"[A-Za-z0-9%°/.-]+")


def words(s): return WORD.findall(s or "")


def duration_days(d):
    """The PATIENT's stated duration, in days, from datasets/<ds>/durations.json.

    Note this is a different thing from the benchmark's `duration_info*` columns, which hold
    per-OPTION disease duration priors (how long each candidate disease typically runs) — those
    are retrieval inputs, not a property of the question."""
    if not isinstance(d, dict): return None
    v = d.get("days")
    # days == 0 is a real hyperacute duration ("today", "just now"), not a missing value — a
    # `> 0` test silently drops those into the "no duration" bucket and understates acute cases
    return float(v) if isinstance(v, (int, float)) and v >= 0 else None


def bucket(days):
    if days is None: return "none"
    for lim, name in [(1, "<1d"), (7, "1-6d"), (30, "1-4wk"),
                      (180, "1-6mo"), (365, "6-12mo")]:
        if days < lim: return name
    return ">1yr"


def critical_uids(ds):
    """(set_of_uids, provenance). Hand review wins over the script when both exist: the script
    only ever perturbs symptom duration, so its verdicts miss the other temporal axes."""
    p = PIPE / f"verification/manual_read_{ds}.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in open(p)]
        return {r["uid"] for r in rows if r["verdict"] == "critical"}, "hand review"
    f = glob.glob(str(PIPE / f"verification/{ds}_duration_critical_*.json"))
    if f:
        rows = json.load(open(f[0]))["results"]
        return {r["uid"] for r in rows if r["verdict"] == "duration_critical"}, "script only"
    return set(), "none"


ORDER = ["<1d", "1-6d", "1-4wk", "1-6mo", "6-12mo", ">1yr", "none"]


def describe(bench, durs, ds):
    """Compute the whole statistics block for one (possibly filtered) list of questions."""
    if not bench: return None
    qlen_c = [len(b["question"]) for b in bench]
    qlen_w = [len(words(b["question"])) for b in bench]
    nopt = [len(b["options"]) for b in bench]
    optlen = [len(words(v)) for b in bench for v in b["options"].values()]
    anslen = [len(words(b["options"][b["answer"]])) for b in bench if b["answer"] in b["options"]]
    vocab = {w.lower() for b in bench for w in words(b["question"])}
    dinfo = [durs.get(b["uid"]) or {} for b in bench]
    days = [duration_days(d) for d in dinfo]
    withdur = [d for d in days if d is not None]

    def avg_len(fn, *keys):
        """Mean list length over THIS subset. These files are {uid: {key: [...]}}, so the list has
        to be reached through its key — len() on the wrapper dict just counts keys."""
        p = PIPE / f"datasets/{ds}/{fn}"
        if not p.exists(): return None
        raw = json.load(open(p))
        n = [sum(len((raw.get(b["uid"]) or {}).get(k) or []) for k in keys)
             for b in bench if b["uid"] in raw]
        return statistics.fmean(n) if n else None

    return {
        "n_questions": len(bench),
        "avg_question_chars": statistics.fmean(qlen_c),
        "median_question_chars": statistics.median(qlen_c),
        "avg_question_words": statistics.fmean(qlen_w),
        "min_question_words": min(qlen_w), "max_question_words": max(qlen_w),
        "avg_n_options": statistics.fmean(nopt),
        "avg_option_words": statistics.fmean(optlen),
        "avg_answer_words": statistics.fmean(anslen),
        "vocabulary_size": len(vocab),
        "n_with_numeric_duration": len(withdur),
        "pct_with_numeric_duration": 100 * len(withdur) / len(bench),
        "median_duration_days": statistics.median(withdur) if withdur else None,
        "avg_n_symptoms": avg_len("symptoms.json", "symptoms"),
        "avg_n_seeds": avg_len("seeds.json", "seeds"),
        "avg_n_query_entities": avg_len("query_entities.json", "symptoms_signs",
                                        "diseases_mentioned", "procedures", "drugs",
                                        "lab_findings"),
        "answer_distribution": {k: round(100 * v / len(bench), 1)
                                for k, v in sorted(Counter(b["answer"] for b in bench).items())},
        "duration_buckets": {k: Counter(bucket(d) for d in days).get(k, 0) for k in ORDER},
        "temporal_role": dict(Counter(d.get("role") for d in dinfo).most_common()),
    }


out = {}
for ds in DATASETS:
    bench = json.load(open(PIPE / f"datasets/{ds}/benchmark.json"))
    dpath = PIPE / f"datasets/{ds}/durations.json"
    durs = json.load(open(dpath)) if dpath.exists() else {}
    crit, prov = critical_uids(ds)
    out[ds] = {"provenance": prov,
               "all": describe(bench, durs, ds),
               "critical": describe([b for b in bench if b["uid"] in crit], durs, ds)}

ROWS = [
    ("# Questions",                     "n_questions",              "{:d}"),
    ("Avg. question length (chars)",    "avg_question_chars",       "{:.1f}"),
    ("Median question length (chars)",  "median_question_chars",    "{:.0f}"),
    ("Avg. question length (words)",    "avg_question_words",       "{:.1f}"),
    ("Avg. # options",                  "avg_n_options",            "{:.2f}"),
    ("Avg. option length (words)",      "avg_option_words",         "{:.1f}"),
    ("Avg. answer length (words)",      "avg_answer_words",         "{:.1f}"),
    ("Vocabulary size (unique words)",  "vocabulary_size",          "{:d}"),
    ("# with numeric duration",         "n_with_numeric_duration",  "{:d}"),
    ("% with numeric duration",         "pct_with_numeric_duration","{:.1f}"),
    ("Median duration (days)",          "median_duration_days",     "{:.1f}"),
    ("Avg. # extracted symptoms",       "avg_n_symptoms",           "{:.2f}"),
    ("Avg. # LLM-DDx seeds",            "avg_n_seeds",              "{:.2f}"),
    ("Avg. # query entities",           "avg_n_query_entities",     "{:.2f}"),
]


def block(kind, title):
    cols = [d for d in DATASETS if out[d][kind]]
    if not cols: return
    w = max(len(r[0]) for r in ROWS) + 2
    print(f"\n{title}")
    print(f"{'':{w}}" + "".join(f"{d:>12}" for d in cols))
    print("-" * (w + 12 * len(cols)))
    for label, key, fmt in ROWS:
        cells = "".join(f"{(fmt.format(out[d][kind][key]) if out[d][kind][key] is not None else '—'):>12}"
                        for d in cols)
        print(f"{label:{w}}{cells}")
    print(f"\n{'Duration bucket':{w}}" + "".join(f"{d:>12}" for d in cols))
    print("-" * (w + 12 * len(cols)))
    for b in ORDER:
        cells = "".join(f"{out[d][kind]['duration_buckets'][b]:>7d} "
                        f"({100*out[d][kind]['duration_buckets'][b]/out[d][kind]['n_questions']:3.0f}%)"
                        for d in cols)
        print(f"{b:{w}}{cells}")


block("all", "=== GENERAL (all questions) ===")
block("critical", "=== DURATION-CRITICAL subset ===")

print("\nsubset provenance: " + "  ".join(f"{d}={out[d]['provenance']}" for d in DATASETS))
for d in DATASETS:
    a, c = out[d]["all"], out[d]["critical"]
    if not c: continue
    print(f"\n{d}: critical {c['n_questions']}/{a['n_questions']} = "
          f"{100*c['n_questions']/a['n_questions']:.1f}% of the set")
    print(f"   answer distribution  all: " +
          "  ".join(f"{k}:{v}" for k, v in a["answer_distribution"].items()))
    print(f"                   critical: " +
          "  ".join(f"{k}:{v}" for k, v in c["answer_distribution"].items()))
    print(f"   temporal role    all: " + "  ".join(f"{k}:{v}" for k, v in a["temporal_role"].items()))
    print(f"                critical: " + "  ".join(f"{k}:{v}" for k, v in c["temporal_role"].items()))

json.dump(out, open(PIPE / "datasets/statistics.json", "w"), indent=1)
print(f"\nsaved -> {PIPE/'datasets/statistics.json'}")
