"""Evaluate Nightmare predictions using the notebook's matching and metrics."""

import csv
import json
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

QUOTE_CHARS = set("\"'`“”‘’«»‹›„‟")


def normalise(text):
    return "".join(ch for ch in text if not ch.isspace() and ch not in QUOTE_CHARS)


def load_gold_strings(ann_path, samples_dir):
    samples = sorted(Path(samples_dir).glob("*.txt"))
    if not samples:
        raise ValueError(f"No sample texts found in {samples_dir}")
    gold = {sample.stem: [] for sample in samples}
    with Path(ann_path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"tagged text", "gold_tag", "text file"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {ann_path}: {required}")
        for row in reader:
            if row["gold_tag"] != "speech":
                continue
            name = Path(row["text file"]).name
            if not name.endswith(".txt") or name[:-4] not in gold:
                raise ValueError(f"Gold annotation references unknown sample: {name}")
            text = normalise(row["tagged text"])
            if text:
                gold[name[:-4]].append(text)
    return gold


def evaluate_book(predicted, gold, threshold):
    if not predicted or not gold:
        counts = {
            "TP": 0, "FP": len(predicted), "FN": len(gold),
            "n_pred": len(predicted), "n_gold": len(gold), "matched_gold": 0,
        }
        return counts, [
            {"string": text, "best_similarity": 0.0, "correct": False}
            for text in predicted
        ]

    scores = cdist(predicted, gold, scorer=fuzz.ratio, dtype=np.float32)
    best_pred_scores = scores.max(axis=1)
    correct = best_pred_scores >= threshold * 100
    tp = int(correct.sum())
    matched_gold = int((scores.max(axis=0) >= threshold * 100).sum())
    counts = {
        "TP": tp, "FP": len(predicted) - tp, "FN": len(gold) - matched_gold,
        "n_pred": len(predicted), "n_gold": len(gold), "matched_gold": matched_gold,
    }
    detail = [
        {"string": text, "best_similarity": round(float(best_pred_scores[i]) / 100, 4),
         "correct": bool(correct[i])}
        for i, text in enumerate(predicted)
    ]
    return counts, detail


def prf1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def write_report(book_results, totals, totals_gold, empty_gold, threshold, report_path, pipeline):
    col = max((len(book["name"]) for book in book_results), default=40)
    header = (
        f"{'Book':<{col}}  {'Pred':>5} {'Gold':>5} "
        f"{'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6}"
    )
    lines = [
        f"=== {pipeline}: character-level dialogue match (threshold={threshold}) ===",
        "",
        "Dialogue is the positive class. A predicted quote is correct (TP) when its best",
        "character-level similarity to any gold quote in the same novel is >= threshold.",
        "FN = gold quotes with no prediction match at threshold.",
        "",
        "=== Per-Novel Results ===",
        header,
        "-" * len(header),
    ]
    for book in book_results:
        lines.append(
            f"{book['name']:<{col}}  {book['n_pred']:>5} {book['n_gold']:>5} "
            f"{book['TP']:>4} {book['FP']:>4} {book['FN']:>4}  "
            f"{book['precision']:>6.3f} {book['recall']:>6.3f} {book['f1']:>6.3f}"
        )
    p, r, f = prf1(totals["TP"], totals["FP"], totals["FN"])
    pg, rg, fg = prf1(totals_gold["TP"], totals_gold["FP"], totals_gold["FN"])
    lines += [
        "",
        "=== Corpus Totals (micro-averaged over all quotes) ===",
        f"Predicted quotes: {totals['n_pred']}   Gold quotes: {totals['n_gold']}",
        f"TP={totals['TP']}  FP={totals['FP']}  FN={totals['FN']}",
        f"Precision: {p:.3f}   Recall: {r:.3f}   F1: {f:.3f}",
        f"Unmatched gold quotes (missed dialogue): {totals['FN']}",
        "",
        f"=== Excluding {len(empty_gold)} books with empty gold lists ===",
        "(These books have no gold quote strings in the nightmare dictionary,",
        "so predictions there can only add false positives.)",
        f"Books excluded: {', '.join(empty_gold) if empty_gold else 'none'}",
        f"TP={totals_gold['TP']}  FP={totals_gold['FP']}  FN={totals_gold['FN']}",
        f"Precision: {pg:.3f}   Recall: {rg:.3f}   F1: {fg:.3f}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport written to {report_path}")


def run_evaluation(predictions, gold, results_dir, threshold=0.90, pipeline_name="booknlp"):
    missing = sorted(gold.keys() - predictions.keys())
    unexpected = sorted(predictions.keys() - gold.keys())
    if missing or unexpected:
        raise ValueError(f"Sample mismatch: missing predictions={missing}; unexpected={unexpected}")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    results_dir = Path(results_dir)
    per_book_dir = results_dir / "per_book"
    per_book_dir.mkdir(parents=True, exist_ok=True)
    totals = dict.fromkeys(("TP", "FP", "FN", "n_pred", "n_gold"), 0)
    totals_gold = totals.copy()
    empty_gold = []
    book_results = []
    for book in sorted(gold):
        counts, detail = evaluate_book(
            [normalise(text) for text in predictions[book]], gold[book], threshold
        )
        for key in totals:
            totals[key] += counts[key]
            if counts["n_gold"]:
                totals_gold[key] += counts[key]
        if not counts["n_gold"]:
            empty_gold.append(book)
        p, r, f = prf1(counts["TP"], counts["FP"], counts["FN"])
        result = {
            "threshold": threshold, **counts, "precision": p, "recall": r,
            "f1": f, "predictions": detail,
        }
        (per_book_dir / (book + ".json")).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        book_results.append({"name": book, **counts, "precision": p, "recall": r, "f1": f})
    write_report(book_results, totals, totals_gold, empty_gold, threshold,
                 results_dir / "evaluation_results.txt", pipeline_name)
