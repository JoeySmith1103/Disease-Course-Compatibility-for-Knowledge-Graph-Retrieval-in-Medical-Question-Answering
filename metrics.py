#!/usr/bin/env python3
"""Classification metrics for multiple-choice reader runs.

WHAT THESE MEASURE — read this before putting a number in a table.

The "classes" here are ANSWER LETTERS, not diagnoses. Option A in one question has nothing to do
with option A in the next. So macro precision/recall over letters is a measure of the reader's
LETTER/POSITION BIAS, not of diagnostic quality: a model that answers D too often is punished, a
model that spreads its answers like the gold distribution is rewarded, and neither fact says
anything about whether it understood the medicine.

Report them as such. In particular do not put them next to retrieval recall (does the gold concept
appear in the retrieved KG pool) — that metric shares a name and measures something entirely
different, and a table holding both under one "recall" heading will be read wrong.

DEFINITIONS, and why they differ from the sketch in new_metrics.py:

  accuracy          correct / n. Unparseable counts as wrong. Equals micro-recall.
  micro_precision   correct / (n − unparseable). Accuracy over the answers that could be parsed;
                    the gap between it and accuracy is exactly the parse-failure cost.
  macro_precision   mean over classes of tp/(tp+fp)
  macro_recall      mean over classes of tp/(tp+fn)   (= balanced accuracy)
  macro_f1          mean over classes of per-class F1  (NOT the F1 of the two macros above)

  Both macros run over the SAME fixed label set with zero_division=0. The sketch skipped classes
  with no predictions when averaging precision but kept them when averaging recall, which averages
  the two over different denominators: precision comes out inflated (a letter the model never
  picks is silently dropped instead of scoring 0) and no F1 built from them is meaningful.

  The label set is FIXED from the dataset's own options, not derived from what happened to be
  observed. Deriving it per run makes macro numbers incomparable across methods, which is the one
  comparison they exist to support.

  macro_* over letters with tiny support is close to meaningless — on 329 two letters have exactly
  one gold question each, so a single item swings 1/10 of the macro. per_class support is always
  returned, `min_support` is reported, and `weighted_*` (support-weighted) is provided as the
  safer headline for imbalanced label sets.

Unparseable answers are FN for their gold class and FP for nothing, so they depress recall without
touching precision — which is the honest treatment: the model failed to answer, it did not answer
wrongly.
"""
import statistics
from collections import Counter


def _labels_from_options(bench_items):
    """Fixed label set = every option letter the dataset actually offers."""
    out = set()
    for b in bench_items:
        out |= set(b.get("options", {}).keys())
    return sorted(out)


def compute_metrics(rows, extract_letter=None, labels=None):
    """One run's metrics.

    rows: dicts with `gold` and either a stored `predicted` or a `raw_response` to re-extract from.
    extract_letter: optional re-extractor. Passing it recomputes predictions from the raw text
      instead of trusting the stored verdict, which is what makes a stored result auditable after
      the extraction rules change.
    labels: fixed class list. Falls back to the observed letters, with the comparability caveat
      above — pass it whenever the numbers will be compared across methods.
    """
    preds, golds = [], []
    for r in rows:
        g = (r.get("gold") or "").upper() or None
        if extract_letter is not None and r.get("raw_response") is not None:
            p = extract_letter(r.get("raw_response") or "")
        else:
            p = r.get("predicted")
        preds.append(p.upper() if isinstance(p, str) else None)
        golds.append(g)

    n = len(rows)
    correct = sum(p is not None and p == g for p, g in zip(preds, golds))
    unparseable = sum(p is None for p in preds)
    parseable = n - unparseable

    if labels is None:
        labels = sorted({g for g in golds if g} | {p for p in preds if p})

    per_class, P, R, F, W = {}, [], [], [], []
    for lab in labels:
        tp = sum(p == lab and g == lab for p, g in zip(preds, golds))
        fp = sum(p == lab and g != lab for p, g in zip(preds, golds))
        fn = sum(p != lab and g == lab for p, g in zip(preds, golds))
        support = tp + fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"tp": tp, "fp": fp, "fn": fn, "support": support,
                          "n_pred": tp + fp,
                          "precision": 100 * prec, "recall": 100 * rec, "f1": 100 * f1}
        # average over classes the DATASET uses (support > 0). A label offered by no question is
        # not a class the model failed at; a label the model never picks is, and scores 0.
        if support:
            P.append(prec); R.append(rec); F.append(f1); W.append(support)

    mean = lambda v: statistics.fmean(v) if v else 0.0
    wmean = lambda v: (sum(x * w for x, w in zip(v, W)) / sum(W)) if W and sum(W) else 0.0
    return {
        "n": n, "correct": correct, "unparseable": unparseable,
        "accuracy":        100 * correct / n if n else 0.0,
        # same quantity under both names: `parseable_precision` is what new_metrics.py called it,
        # `micro_precision` is what it is (micro-averaged precision over a single-label problem).
        # Kept as an alias rather than renamed, so existing notes and this file agree.
        "micro_precision":     100 * correct / parseable if parseable else 0.0,
        "parseable_precision": 100 * correct / parseable if parseable else 0.0,
        "macro_precision": 100 * mean(P),
        "macro_recall":    100 * mean(R),
        "macro_f1":        100 * mean(F),
        "weighted_precision": 100 * wmean(P),
        "weighted_recall":    100 * wmean(R),
        "weighted_f1":        100 * wmean(F),
        "n_classes": len(W),
        "min_support": min(W) if W else 0,
        "per_class": per_class,
        # letter bias: how far the answer distribution drifts from the gold distribution.
        # 0 = identical, 100 = disjoint. Reported because macro_recall moves with it and the
        # cause is otherwise invisible.
        "letter_bias_tvd": _tvd(preds, golds),
        "pred_dist": dict(sorted(Counter(p for p in preds if p).items())),
        "gold_dist": dict(sorted(Counter(g for g in golds if g).items())),
    }


def _tvd(preds, golds):
    """Total variation distance (%) between the predicted and gold letter distributions."""
    pc, gc = Counter(p for p in preds if p), Counter(g for g in golds if g)
    np_, ng = sum(pc.values()), sum(gc.values())
    if not np_ or not ng:
        return 0.0
    keys = set(pc) | set(gc)
    return 50 * sum(abs(pc[k] / np_ - gc[k] / ng) for k in keys)


AGG_KEYS = ["accuracy", "micro_precision", "parseable_precision",
            "macro_precision", "macro_recall", "macro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1", "letter_bias_tvd"]


def aggregate(run_metrics):
    """mean ± population std across runs, matching how accuracy is already reported."""
    out = {}
    for k in AGG_KEYS:
        v = [m[k] for m in run_metrics if k in m]
        out[k] = statistics.fmean(v) if v else 0.0
        out[k + "_std"] = statistics.pstdev(v) if len(v) > 1 else 0.0
    out["runs_" + "accuracy"] = [m["accuracy"] for m in run_metrics]
    out["n"] = run_metrics[0]["n"] if run_metrics else 0
    out["n_runs"] = len(run_metrics)
    out["unparseable"] = [m["unparseable"] for m in run_metrics]
    out["min_support"] = min((m["min_support"] for m in run_metrics), default=0)
    out["n_classes"] = run_metrics[0]["n_classes"] if run_metrics else 0
    return out


def format_report(agg, title="", warn_support=5):
    L = []
    if title:
        L.append(title)
    L.append(f"  n={agg['n']}  runs={agg['n_runs']}  unparseable={agg['unparseable']}  "
             f"classes={agg['n_classes']} (min support {agg['min_support']})")
    rows = [("accuracy",           "Accuracy (= micro-recall)"),
            ("parseable_precision", "Parseable-precision (= micro-P)"),
            ("macro_recall",       "Macro-recall (= balanced acc)"),
            ("macro_precision",    "Macro-precision"),
            ("macro_f1",           "Macro-F1"),
            ("weighted_recall",    "Weighted-recall"),
            ("weighted_precision", "Weighted-precision"),
            ("weighted_f1",        "Weighted-F1"),
            ("letter_bias_tvd",    "Letter-bias TVD (0=無偏)")]
    for k, lab in rows:
        L.append(f"    {lab:34s} {agg[k]:6.2f} ± {agg[k+'_std']:.2f}")
    if agg["min_support"] < warn_support:
        L.append(f"    ⚠ 最小類別只有 {agg['min_support']} 題 — macro_* 由極少數題目主導，"
                 f"改看 weighted_*")
    return "\n".join(L)
