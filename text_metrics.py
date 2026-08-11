#!/usr/bin/env python3
"""Token-overlap precision / recall between the model's raw output and the gold option's TEXT.

For each question:
    gold   = options[gold_letter]          e.g. "Perform liver mass resection"
    pred   = the model's raw_response, in full
    overlap = multiset intersection of their tokens
    precision = overlap / |pred tokens|
    recall    = overlap / |gold tokens|

Multisets (Counter &), so a word repeated in the output cannot be credited twice against a gold
token that occurs once.

Tokens are lowercased word characters with a short stopword list removed. The list holds only
words that cannot distinguish two options — matching on "the"/"of" would make every option pair
look alike. Anything clinical (dose, route, laterality, timing) stays in.

Both numbers move with output LENGTH: precision is bounded above by |gold|/|pred|, so a long
reasoning block scores low precision no matter how right it is. Compare across runs of the same
prompt, not across prompts that ask for different amounts of text.

Usage:  python3 pipeline/text_metrics.py [dataset ...]      # default: medbullets
"""
import re
from collections import Counter

STOP = {"a", "an", "the", "of", "to", "in", "for", "with", "and", "or", "on", "at", "by",
        "is", "are", "was", "were", "be", "this", "that", "it", "as", "from"}
_WORD = re.compile(r"[a-z0-9]+")


def tokens(text, drop_stop=True):
    t = _WORD.findall((text or "").lower())
    return [w for w in t if not (drop_stop and w in STOP)]


def prf(pred_text, gold_text, drop_stop=True):
    """(precision, recall, f1) in 0..1 for one question."""
    p_tok, g_tok = tokens(pred_text, drop_stop), tokens(gold_text, drop_stop)
    if not p_tok or not g_tok:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(p_tok) & Counter(g_tok)).values())
    p = overlap / len(p_tok)
    r = overlap / len(g_tok)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


if __name__ == "__main__":
    import json, glob, os, statistics, sys

    PIPE = os.path.dirname(os.path.abspath(__file__))
    for ds in (sys.argv[1:] or ["medbullets"]):
        bench = {x["uid"]: x for x in json.load(open(f"{PIPE}/datasets/{ds}/benchmark.json"))}
        for sub in ["old_prompt", "revised_prompt"]:
            files = sorted(glob.glob(f"{PIPE}/results/{sub}/{ds}_*.json"))
            if not files:
                continue
            rows = []
            for f in files:
                d = json.load(open(f))
                per = []
                for run in d["runs"]:
                    s = [prf(r.get("raw_response"),
                             bench[r["uid"]]["options"].get(r["gold"], ""))
                         for r in run["results"]]
                    per.append([100 * statistics.fmean(x[i] for x in s) for i in range(3)])
                rows.append((d["method"].replace("__revised", ""),
                             *[statistics.fmean(x[i] for x in per) for i in range(3)],
                             *[statistics.pstdev(x[i] for x in per) if len(per) > 1 else 0.0
                               for i in range(3)]))
            rows.sort(key=lambda r: -r[2])
            print(f"\n### {ds} · {sub} · raw_output vs 正解選項文字")
            print(f"{'method':18}{'Precision':>18}{'Recall':>18}{'F1':>18}")
            print("-" * 72)
            for m, p, r, f, sp, sr, sf in rows:
                print(f"{m:18}{f'{p:.2f} ± {sp:.2f}':>18}{f'{r:.2f} ± {sr:.2f}':>18}"
                      f"{f'{f:.2f} ± {sf:.2f}':>18}")
