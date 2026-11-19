"""
Shared evaluation core for character-level dialogue detection.

RiQuA corpus version — gold annotations are loaded directly from riqua's
native BRAT standoff .ann files (character-offset spans), not from a
derived format like LitBank's sentence/token QUOTE-tab files or Dialogism's
XML.

This module contains everything that is pipeline-agnostic:

  - normalise()         — reduces a string to its visible, non-quotation characters
  - load_gold_strings() — reads riqua BRAT gold annotations
  - best_similarity()   — fuzzy string matching via RapidFuzz (single-pair helper)
  - evaluate_book()     — scores one novel (TP/FP/FN), via a vectorised RapidFuzz cdist
  - prf1()              — precision / recall / F1 from counts
  - run_evaluation()    — the main loop: pairs predictions with gold, scores, reports
  - write_report()      — formats and writes the summary text file

No loader logic lives here. Loaders are separate modules in loaders/ that each
expose a load_predictions(data_path, ann_dir) function returning:

    {book_name: [raw_dialogue_string, ...], ...}

The book_name keys must correspond to gold .ann files as:
    <ann_dir>/<book_name>_brat.ann

Normalisation happens here (not in loaders) so every pipeline is normalised
identically.
"""

import os
import json
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cdist, extractOne

# ── Normalisation ─────────────────────────────────────────────────────────────

# Characters stripped before comparison: every flavour of straight/curly quote,
# plus all whitespace (handled separately via the isspace() check).
QUOTE_CHARS = set("\"'`“”‘’«»‹›„‟")


def normalise(text):
    """Reduce a string to its visible, non-quotation characters.

    Drops all whitespace (spaces, tabs, newlines) and every quotation mark
    listed in QUOTE_CHARS, so similarity reflects spoken content rather than
    punctuation choices or layout differences between pipelines.

    Examples
    --------
    >>> normalise('"Oh dear!" she said.')
    'Ohdear!shesaid.'
    >>> normalise("  'Oh dear!'  ")
    'Ohdear!'
    """
    return "".join(ch for ch in text if not ch.isspace() and ch not in QUOTE_CHARS)


# ── Gold loading ──────────────────────────────────────────────────────────────

def load_gold_strings(ann_path):
    """Return normalised dialogue strings from a riqua gold .ann file.

    riqua's .ann files are genuine BRAT standoff annotations, e.g.:

        T1  Quotation 11147 11418  "My dearest papa! ..."
        T2  Cue 11053 11057        said
        T6  Entity 16598 16611     Mr. Woodhouse
        R1  Addressee Arg1:T4 Arg2:T1
        R2  Speaker Arg1:T3 Arg2:T1

    Each entity line is tab-separated: id, then "<type> <start> <end>", then
    the covered text. We take every T-line whose type is "Quotation" (both
    direct and indirect quotes are tagged this way in riqua — there is no
    separate indirect-quote type), grab its text field, normalise it, and
    return the list. Cue/Entity T-lines and R- (relation) lines are skipped,
    since only the quotation span itself is comparable to what the pipelines
    under test predict.
    """
    strings = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            span_type = parts[1].split(" ", 1)[0]
            if span_type != "Quotation":
                continue
            norm = normalise(parts[2])
            if norm:
                strings.append(norm)
    return strings


# ── Matching ──────────────────────────────────────────────────────────────────

def best_similarity(needle, haystack):
    """Highest RapidFuzz ratio between *needle* and any string in *haystack*.

    Single-pair convenience helper (used by evaluate_book's edge cases where
    predicted or gold is empty). For the main scoring path, evaluate_book()
    below uses rapidfuzz.process.cdist to score every pair at once instead of
    calling this in a Python loop.
    """
    if not needle or not haystack:
        return 0.0
    result = extractOne(needle, haystack, scorer=fuzz.ratio)
    return result[1] / 100.0 if result else 0.0


def evaluate_book(predicted, gold, threshold):
    """Score one novel. Both `predicted` and `gold` should already be normalised.

    Returns (counts_dict, predicted_detail_list).

    counts_dict has keys: TP, FP, FN, n_pred, n_gold, matched_gold.
    predicted_detail is a list of dicts with {string, best_similarity, correct}
    so that per-book JSON output shows exactly why each prediction passed/failed.

    Scoring logic
    -------------
    - Precision direction: for each predicted string, find its best match in gold.
      If similarity >= threshold, it's a TP; otherwise FP.
    - Recall direction: for each gold string, find its best match in predicted.
      If nothing reaches the threshold, it's an FN (a missed gold quote).

    The two directions are independent — multiple predictions can match the same
    gold string, which is standard for fuzzy set-overlap scoring.

    Uses rapidfuzz.process.cdist to compute the full (n_pred x n_gold) similarity
    matrix in one compiled C++ call — no Python loops over string pairs. Precision
    and recall are both read off the matrix with numpy max operations.
    """
    if not predicted and not gold:
        return {"TP": 0, "FP": 0, "FN": 0, "n_pred": 0, "n_gold": 0,
                "matched_gold": 0}, []

    if not predicted:
        return {"TP": 0, "FP": 0, "FN": len(gold), "n_pred": 0,
                "n_gold": len(gold), "matched_gold": 0}, []

    if not gold:
        pred_detail = [{"string": p, "best_similarity": 0.0, "correct": False}
                       for p in predicted]
        return {"TP": 0, "FP": len(predicted), "FN": 0, "n_pred": len(predicted),
                "n_gold": 0, "matched_gold": 0}, pred_detail

    thresh_pct = threshold * 100  # rapidfuzz scores are 0-100

    # One compiled call computes every (predicted[i], gold[j]) pair.
    # Shape: (n_pred, n_gold), dtype float32, values 0-100.
    scores = cdist(predicted, gold, scorer=fuzz.ratio, dtype=np.float32)

    # ── Precision ─────────────────────────────────────────────────────────────
    best_pred_scores = scores.max(axis=1)          # shape (n_pred,)
    correct_mask     = best_pred_scores >= thresh_pct
    tp = int(correct_mask.sum())
    fp = len(predicted) - tp

    # ── Recall ────────────────────────────────────────────────────────────────
    best_gold_scores = scores.max(axis=0)          # shape (n_gold,)
    matched_gold     = int((best_gold_scores >= thresh_pct).sum())
    fn = len(gold) - matched_gold

    pred_detail = [
        {
            "string":          p,
            "best_similarity": round(float(best_pred_scores[i]) / 100, 4),
            "correct":         bool(correct_mask[i]),
        }
        for i, p in enumerate(predicted)
    ]

    counts = {
        "TP": tp, "FP": fp, "FN": fn,
        "n_pred": len(predicted), "n_gold": len(gold),
        "matched_gold": matched_gold,
    }
    return counts, pred_detail


def prf1(tp, fp, fn):
    """Compute precision, recall, and F1 from raw counts.

    Returns (0, 0, 0) when denominators are zero, which happens when a book
    has no predictions or no gold annotations.
    """
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(predictions, ann_dir, results_dir, threshold=0.90,
                   pipeline_name="pipeline"):
    """Run the full evaluation given a predictions dict and gold directory.

    Parameters
    ----------
    predictions : dict[str, list[str]]
        {book_name: [raw_dialogue_string, ...]} as returned by any loader.
        Strings will be normalised here.
    ann_dir : str
        Path to the directory containing <book_name>_brat.ann gold files.
    results_dir : str
        Where to write results. Will create per_book/ inside it.
    threshold : float
        Similarity at or above which a predicted string counts as matching.
    pipeline_name : str
        Label used in the report header (e.g. "booknlp").
    """
    per_book_dir = os.path.join(results_dir, "per_book")
    os.makedirs(per_book_dir, exist_ok=True)

    book_results = []
    tot = {"TP": 0, "FP": 0, "FN": 0, "n_pred": 0, "n_gold": 0}

    for book_name in sorted(predictions.keys()):
        ann_path = os.path.join(ann_dir, book_name + "_brat.ann")
        if not os.path.exists(ann_path):
            print(f"Skipping {book_name}: no gold .ann ({book_name}_brat.ann)")
            continue

        # Normalise the raw predicted strings (gold is normalised by load_gold_strings)
        predicted_norm = [normalise(s) for s in predictions[book_name]]
        predicted_norm = [s for s in predicted_norm if s]  # drop empties

        gold = load_gold_strings(ann_path)

        counts, pred_detail = evaluate_book(predicted_norm, gold, threshold)

        # Accumulate totals
        for k in tot:
            tot[k] += counts[k]

        p, r, f = prf1(counts["TP"], counts["FP"], counts["FN"])
        book_results.append({
            "name": book_name, **counts,
            "precision": p, "recall": r, "f1": f,
        })
        print(
            f"Evaluated {book_name}: "
            f"pred={counts['n_pred']} gold={counts['n_gold']} "
            f"TP={counts['TP']} FP={counts['FP']} FN={counts['FN']} "
            f"F1={f:.3f}"
        )

        # Write per-book detail JSON
        with open(os.path.join(per_book_dir, book_name + ".json"), "w",
                  encoding="utf-8") as fh:
            json.dump(
                {"threshold": threshold, **counts,
                 "precision": p, "recall": r, "f1": f,
                 "predictions": pred_detail},
                fh, indent=2, ensure_ascii=False,
            )

    report_path = os.path.join(results_dir, "evaluation_results.txt")
    write_report(book_results, tot, threshold, report_path, pipeline_name)


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(book_results, tot, threshold, report_path, pipeline_name):
    """Format and write the human-readable evaluation report."""
    book_results.sort(key=lambda b: b["name"])
    col = max((len(b["name"]) for b in book_results), default=40)

    header = (
        f"{'Book':<{col}}  {'Pred':>5} {'Gold':>5} "
        f"{'TP':>4} {'FP':>4} {'FN':>4}  "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6}"
    )
    lines = [
        f"=== {pipeline_name}: character-level dialogue match "
        f"(threshold={threshold}) — RiQuA corpus (merged gold) ===",
        "",
        "Dialogue is the positive class. A predicted quote is correct (TP) "
        "when its best character-level",
        "similarity to any gold quote in the same novel is >= threshold. "
        "FN = gold quotes nothing matched.",
        "",
        "=== Per-Novel Results ===",
        header,
        "-" * len(header),
    ]
    for b in book_results:
        lines.append(
            f"{b['name']:<{col}}  {b['n_pred']:>5} {b['n_gold']:>5} "
            f"{b['TP']:>4} {b['FP']:>4} {b['FN']:>4}  "
            f"{b['precision']:>6.3f} {b['recall']:>6.3f} {b['f1']:>6.3f}"
        )

    p, r, f = prf1(tot["TP"], tot["FP"], tot["FN"])

    lines += [
        "",
        "=== Corpus Totals (micro-averaged over all quotes) ===",
        f"Predicted quotes: {tot['n_pred']}   Gold quotes: {tot['n_gold']}",
        f"TP={tot['TP']}  FP={tot['FP']}  FN={tot['FN']}",
        f"Precision: {p:.3f}   Recall: {r:.3f}   F1: {f:.3f}",
        f"Unmatched gold quotes (missed dialogue): {tot['FN']}",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nReport written to {report_path}")
