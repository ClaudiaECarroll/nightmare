"""
Shared evaluation core for character-level dialogue detection.
Dialogism corpus version — gold annotations are loaded from XML, not .ann files.

This module contains everything that is pipeline-agnostic:

  - normalise()                  — reduces a string to visible, non-quotation chars
  - load_gold_strings_from_xml() — reads Dialogism XML gold annotations
  - load_gold_strings()          — reads LitBank .ann gold annotations (kept for reference)
  - find_xml_file()              — locates the XML file for a given book name
  - best_similarity()            — fuzzy string matching via SequenceMatcher
  - evaluate_book()              — scores one novel (TP/FP/FN)
  - prf1()                       — precision / recall / F1 from counts
  - run_evaluation()             — the main loop: pairs predictions with gold, scores, reports
  - write_report()               — formats and writes the summary text file

No loader logic lives here. Loaders are separate modules in loaders/ that each
expose a load_predictions(data_path, ann_dir) function returning:

    {book_name: [raw_dialogue_string, ...], ...}

The book_name keys must match the Dialogism XML files as:
    <gold_dir>/<book_name>_annotated[_newlinesreduced].xml

Normalisation happens here (not in loaders) so every pipeline is normalised
identically.
"""

import os
import re
import glob
import json
from difflib import SequenceMatcher
from rapidfuzz import fuzz
from rapidfuzz.process import cdist
import numpy as np

# ── Normalisation ─────────────────────────────────────────────────────────────

QUOTE_CHARS = set("\"'`“”‘’«»‹›„‟")


def normalise(text):
    """Reduce a string to its visible, non-quotation characters.

    Drops all whitespace and every quotation mark listed in QUOTE_CHARS, so
    similarity reflects spoken content rather than punctuation or layout.

    >>> normalise('"Oh dear!" she said.')
    'Ohdear!shesaid.'
    """
    return "".join(ch for ch in text if not ch.isspace() and ch not in QUOTE_CHARS)


# ── Gold loading ──────────────────────────────────────────────────────────────

def find_xml_file(xml_dir, book_name):
    """Return the path to the Dialogism XML file for book_name, or None.

    Looks for <xml_dir>/<book_name>_annotated*.xml to handle the
    _newlinesreduced variant in addition to the plain _annotated form.
    """
    pattern = os.path.join(xml_dir, book_name + "_annotated*.xml")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_gold_strings_from_xml(xml_path):
    """Return normalised dialogue strings from a Dialogism annotated XML file.

    Extracts text content from <quote> elements, skipping those with
    speaker="NONE" (non-dialogue quoted text such as newspaper headlines or
    book titles cited by the narrator).

    Quote bodies may contain nested <mention ...>...</mention> coreference
    tags and HTML-entity-encoded quotation marks (&quot;) in some files; both
    are cleaned up before normalising.

    The raw text is normalised with normalise() before being returned, so the
    list can be compared directly against normalised predictions.
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # speaker="..." may appear anywhere among the tag's attributes, not just first.
    # Non-greedy match handles multi-line quotes and adjacent quote tags correctly.
    pattern = re.compile(r'<quote[^>]*\bspeaker="([^"]*)"[^>]*>(.*?)</quote>', re.DOTALL)
    strings = []
    for speaker, text in pattern.findall(raw):
        if speaker == "NONE":
            continue
        text = re.sub(r"<mention[^>]*>", "", text)
        text = re.sub(r"</mention>", "", text)
        text = text.replace("&quot;", '"')
        norm = normalise(text)
        if norm:
            strings.append(norm)
    return strings


def load_gold_strings(ann_path):
    """Return normalised dialogue strings from a LitBank .ann file.

    Kept for reference. Not used by run_evaluation() in this Dialogism copy;
    gold is loaded via load_gold_strings_from_xml() instead.
    """
    strings = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or parts[0] != "QUOTE" or len(parts) < 7:
                continue
            norm = normalise(parts[6])
            if norm:
                strings.append(norm)
    return strings


# ── Matching ──────────────────────────────────────────────────────────────────

def best_similarity(needle, haystack):
    """Highest similarity ratio between needle and any string in haystack."""
    if not needle or not haystack:
        return 0.0
    from rapidfuzz.process import extractOne
    result = extractOne(needle, haystack, scorer=fuzz.ratio)
    return result[1] / 100.0 if result else 0.0


def evaluate_book(predicted, gold, threshold):
    """Score one novel. Both predicted and gold should already be normalised.

    Returns (counts_dict, predicted_detail_list).

    counts_dict has keys: TP, FP, FN, n_pred, n_gold, matched_gold.
    predicted_detail is a list of dicts with {string, best_similarity, correct}.

    Uses rapidfuzz.process.cdist to compute the full (n_pred × n_gold) similarity
    matrix in one compiled C++ call — no Python loops over string pairs.
    Precision and recall are both read off the matrix with numpy max operations.
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

    thresh_pct = threshold * 100  # rapidfuzz scores are 0–100

    # One compiled call computes every (predicted[i], gold[j]) pair.
    # Shape: (n_pred, n_gold), dtype float32, values 0–100.
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
    """Compute precision, recall, and F1 from raw counts."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(predictions, gold_dir, results_dir, threshold=0.90,
                   pipeline_name="pipeline"):
    """Run the full evaluation given a predictions dict and Dialogism XML directory.

    Parameters
    ----------
    predictions : dict[str, list[str]]
        {book_name: [raw_dialogue_string, ...]} as returned by any loader.
        Strings will be normalised here.
    gold_dir : str
        Path to the directory containing Dialogism <book_name>_annotated*.xml files.
    results_dir : str
        Where to write results. Creates per_book/ inside it automatically.
    threshold : float
        Similarity at or above which a predicted string counts as matching.
    pipeline_name : str
        Label used in the report header.
    """
    per_book_dir = os.path.join(results_dir, "per_book")
    os.makedirs(per_book_dir, exist_ok=True)

    book_results = []
    tot = {"TP": 0, "FP": 0, "FN": 0, "n_pred": 0, "n_gold": 0}

    for book_name in sorted(predictions.keys()):
        xml_path = find_xml_file(gold_dir, book_name)
        if xml_path is None:
            print(f"Skipping {book_name}: no gold XML found in {gold_dir}")
            continue

        predicted_norm = [normalise(s) for s in predictions[book_name]]
        predicted_norm = [s for s in predicted_norm if s]

        gold = load_gold_strings_from_xml(xml_path)

        counts, pred_detail = evaluate_book(predicted_norm, gold, threshold)

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
        f"(threshold={threshold}) — Dialogism corpus ===",
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
