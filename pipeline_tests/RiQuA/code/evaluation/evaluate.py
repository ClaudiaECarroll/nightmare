#!/usr/bin/env python3
"""
Evaluate dialogue extraction accuracy against RiQuA gold annotations.

This is the riqua copy of the LitBank/Dialogism evaluation CLI. Each pipeline
has a loader module in loaders/ that knows how to read its particular output
format and return a standard dict of {book_name: [dialogue_strings]}; those
loader modules are unchanged from LitBank. Gold is loaded from riqua's native
BRAT .ann files (see core.load_gold_strings) instead of LitBank's derived
QUOTE-tab format or Dialogism's XML.

Usage
-----
    # Evaluate BookNLP output (a directory of per-book folders):
    python evaluate.py booknlp \\
        --data-path ../../data/intermediate/BookNLP_output

    # Evaluate QuoteAnnotator output (a directory of per-book JSON files):
    python evaluate.py quoteannotator \\
        --data-path ../../data/intermediate/QuoteAnnotator_output

    # Use a different similarity threshold:
    python evaluate.py booknlp \\
        --data-path ../../data/intermediate/BookNLP_output \\
        --threshold 0.80

    # Tag a run to avoid overwriting previous results:
    python evaluate.py booknlp \\
        --data-path ../../data/intermediate/BookNLP_output \\
        --run-label after_bugfix

Results are written to:
    ../../data/results/<pipeline_name>/evaluation_results.txt   (summary)
    ../../data/results/<pipeline_name>/per_book/<book>.json      (per-book detail)

Or, with --run-label:
    ../../data/results/<pipeline_name>_<run_label>/...
"""

import os
import argparse

from loaders import REGISTRY
from core import run_evaluation

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANN_DIR      = os.path.join(HERE, "..", "..", "data", "raw", "labels")
DEFAULT_RESULTS_ROOT = os.path.join(HERE, "..", "..", "data", "results")
DEFAULT_THRESHOLD    = 0.90


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate dialogue extraction against RiQuA gold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Positional: which pipeline to evaluate ──
    ap.add_argument(
        "pipeline",
        choices=sorted(REGISTRY.keys()),
        help="Which extraction pipeline produced the predictions. "
             f"Available: {', '.join(sorted(REGISTRY.keys()))}",
    )

    # ── Required: where are the predictions? ──
    ap.add_argument(
        "--data-path",
        required=True,
        help="Path to the pipeline's output directory of per-book folders/files.",
    )

    # ── Optional ──
    ap.add_argument(
        "--ann-dir",
        default=DEFAULT_ANN_DIR,
        help="Directory containing <book>_brat.ann gold files. "
             f"Default: {DEFAULT_ANN_DIR}",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="SequenceMatcher similarity at or above which a predicted string "
             f"counts as matching a gold string. Default: {DEFAULT_THRESHOLD}",
    )
    ap.add_argument(
        "--run-label",
        default=None,
        help="Optional tag appended to the results folder name, so you can "
             "keep multiple runs without overwriting. "
             "e.g. --run-label v2 → results/booknlp_v2/",
    )
    ap.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Parent directory for results. A subdirectory per pipeline "
             f"is created inside it. Default: {DEFAULT_RESULTS_ROOT}",
    )

    args = ap.parse_args()

    # ── Resolve the loader module ──
    loader = REGISTRY[args.pipeline]

    # ── Build the results directory name ──
    results_name = args.pipeline
    if args.run_label:
        results_name += "_" + args.run_label
    results_dir = os.path.join(args.results_root, results_name)

    # ── Load predictions ──
    print(f"Loading predictions from: {args.data_path}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Gold annotations: {args.ann_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Results will be written to: {results_dir}")
    print()

    predictions = loader.load_predictions(args.data_path, args.ann_dir)

    if not predictions:
        print("ERROR: loader returned no predictions. Check --data-path.")
        return

    print(f"Loaded predictions for {len(predictions)} book(s).\n")

    # ── Run evaluation ──
    run_evaluation(
        predictions=predictions,
        ann_dir=args.ann_dir,
        results_dir=results_dir,
        threshold=args.threshold,
        pipeline_name=args.pipeline,
    )


if __name__ == "__main__":
    main()
