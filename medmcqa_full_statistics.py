#!/usr/bin/env python3
"""ReinRAG-style statistics for the ORIGINAL, unfiltered MedMCQA — all three splits.

The thesis only reviews 143 MedMCQA questions, which is 3.4% of the validation split. Any rate
measured on those 143 is therefore a rate on a heavily enriched slice, and reporting it without
the parent distribution invites the reader to mistake it for a property of MedMCQA. This script
produces the parent distribution: what the dataset looks like before any of our filtering.

The parquet files are pulled from the HF parquet endpoint (85 MB for train), so they are cached
under /tmp rather than committed. Re-download with:
  curl -sL "https://huggingface.co/api/datasets/openlifescienceai/medmcqa/parquet/default/train"
  # -> returns the URL of 0.parquet

Usage:  python3 pipeline/medmcqa_full_statistics.py
Output: table + pipeline/datasets/medmcqa_full_statistics.json
"""
import json, re, statistics
from pathlib import Path
from collections import Counter

import pandas as pd

PIPE = Path(__file__).resolve().parent
WORD = re.compile(r"[A-Za-z0-9%°/.-]+")
SRC = {"train": "/tmp/mmc_train.parquet",
       "validation": "/tmp/mmc_validation.parquet",
       "test": "/tmp/mmc_test.parquet"}
# validation was downloaded through the rows endpoint earlier; reuse that cache if the parquet
# is absent so the script works without a second large download
VAL_JSON = PIPE / "datasets/_raw_medmcqa_validation.json"

# non-capturing groups: pandas .str.contains warns (and may misbehave) on capture groups
TEMPORAL = re.compile(r"\b\d+\s*[- ]?(?:hour|day|week|month|year|minute)s?\b|"
                      r"\b(?:since|ago|for the (?:past|last)|history of|onset|duration|lasting)\b", re.I)
VIGNETTE = re.compile(r"\b(?:\d+[- ]?(?:year|month|day|week)[- ]?old|patient|presents|presented|"
                      r"man|woman|boy|girl|complains)\b", re.I)


def load(split):
    p = Path(SRC[split])
    if p.exists():
        return pd.read_parquet(p)
    if split == "validation" and VAL_JSON.exists():
        return pd.DataFrame(json.load(open(VAL_JSON)))
    return None


def describe(df, split):
    q = df["question"].astype(str)
    # `cop` is 0-indexed in the HF release; the official test split ships without answers
    labelled = df["cop"].between(0, 3).sum() if "cop" in df else 0
    chars = q.str.len()
    wordn = q.map(lambda s: len(WORD.findall(s)))
    optcols = [c for c in ["opa", "opb", "opc", "opd"] if c in df]
    optw = pd.concat([df[c].astype(str).map(lambda s: len(WORD.findall(s))) for c in optcols])
    vocab = set()
    for s in q: vocab.update(w.lower() for w in WORD.findall(s))
    vig = q.str.contains(VIGNETTE, regex=True, na=False)
    tmp = q.str.contains(TEMPORAL, regex=True, na=False)
    long = chars >= 200
    return {
        "split": split,
        "n_questions": int(len(df)),
        "n_labelled": int(labelled),
        "pct_labelled": round(100 * labelled / len(df), 1),
        "avg_question_chars": round(chars.mean(), 1),
        "median_question_chars": int(chars.median()),
        "p90_question_chars": int(chars.quantile(0.90)),
        "max_question_chars": int(chars.max()),
        "avg_question_words": round(wordn.mean(), 1),
        "median_question_words": int(wordn.median()),
        "n_options": len(optcols),
        "avg_option_words": round(optw.mean(), 1),
        "vocabulary_size": len(vocab),
        "len_buckets": {"<100": int((chars < 100).sum()), "100-199": int(((chars >= 100) & (chars < 200)).sum()),
                        "200-399": int(((chars >= 200) & (chars < 400)).sum()),
                        ">=400": int((chars >= 400).sum())},
        "n_vignette": int(vig.sum()),
        "n_temporal": int(tmp.sum()),
        "n_len200": int(long.sum()),
        "n_vignette_len200": int((vig & long).sum()),
        "n_vignette_len200_temporal": int((vig & long & tmp).sum()),
        "answer_distribution": ({str(k): round(100 * v / labelled, 1)
                                 for k, v in sorted(Counter(df.loc[df["cop"].between(0, 3), "cop"]).items())}
                                if labelled else {}),
        "subjects": dict(Counter(df["subject_name"]).most_common(8)) if "subject_name" in df else {},
    }


out = {}
for split in ["train", "validation", "test"]:
    df = load(split)
    if df is None:
        print(f"[skip] {split}: no local parquet/json — see the docstring for the download command")
        continue
    out[split] = describe(df, split)

ROWS = [("# Questions", "n_questions", "{:,}"), ("# with answer label", "n_labelled", "{:,}"),
        ("% labelled", "pct_labelled", "{:.1f}"),
        ("Avg. question length (chars)", "avg_question_chars", "{:.1f}"),
        ("Median question length (chars)", "median_question_chars", "{:,}"),
        ("P90 question length (chars)", "p90_question_chars", "{:,}"),
        ("Max question length (chars)", "max_question_chars", "{:,}"),
        ("Avg. question length (words)", "avg_question_words", "{:.1f}"),
        ("Median question length (words)", "median_question_words", "{:,}"),
        ("# options", "n_options", "{:d}"),
        ("Avg. option length (words)", "avg_option_words", "{:.1f}"),
        ("Vocabulary size", "vocabulary_size", "{:,}")]

cols = list(out)
w = max(len(r[0]) for r in ROWS) + 2
print(f"\n=== MedMCQA — ORIGINAL dataset, all splits (no filtering) ===\n")
print(f"{'':{w}}" + "".join(f"{c:>14}" for c in cols))
print("-" * (w + 14 * len(cols)))
for label, key, fmt in ROWS:
    print(f"{label:{w}}" + "".join(f"{fmt.format(out[c][key]):>14}" for c in cols))

print(f"\n{'Question length':{w}}" + "".join(f"{c:>14}" for c in cols))
print("-" * (w + 14 * len(cols)))
for b in ["<100", "100-199", "200-399", ">=400"]:
    print(f"{b:{w}}" + "".join(f"{out[c]['len_buckets'][b]:>8,} ({100*out[c]['len_buckets'][b]/out[c]['n_questions']:3.0f}%)"
                               for c in cols))

print(f"\n{'Screening funnel':{w}}" + "".join(f"{c:>14}" for c in cols))
print("-" * (w + 14 * len(cols)))
for label, key in [("all questions", "n_questions"), ("+ >=200 chars", "n_len200"),
                   ("+ vignette wording", "n_vignette_len200"),
                   ("+ temporal expression", "n_vignette_len200_temporal")]:
    print(f"{label:{w}}" + "".join(f"{out[c][key]:>8,} ({100*out[c][key]/out[c]['n_questions']:3.1f}%)"
                                   for c in cols))

for c in cols:
    if out[c]["subjects"]:
        print(f"\n{c} top subjects: " + "  ".join(f"{k}:{v:,}" for k, v in out[c]["subjects"].items()))

json.dump(out, open(PIPE / "datasets/medmcqa_full_statistics.json", "w"), indent=1)
print(f"\nsaved -> {PIPE/'datasets/medmcqa_full_statistics.json'}")
