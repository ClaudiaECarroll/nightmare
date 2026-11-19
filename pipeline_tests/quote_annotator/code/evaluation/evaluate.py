#!/usr/bin/env python3
"""
Evaluate dialogue extraction accuracy against Dialogism corpus gold annotations.

This is a unified CLI that works with any supported pipeline. Each pipeline
has a loader module in loaders/ that knows how to read its particular output
format and return a standard dict of {book_name: [dialogue_strings]}.

Gold annotations are loaded from Dialogism XML files
(<gold_dir>/<book_name>_annotated[_newlinesreduced].xml).

NOTE: Pipeline loader modules warn "no gold file for <book>" because they look
for .ann files (the LitBank format) and won't find any here. Those warnings are
expected and harmless — the actual gold loading uses the XML files.

Usage
-----
    # Evaluate BookNLP output (a directory of per-book folders):
    python evaluate.py booknlp \\
        --data-path ../../data/intermediate/BookNLP_output

    # Use a different similarity threshold:
    python evaluate.py booknlp \\
        --data-path ../../data/intermediate/BookNLP_output \\
        --threshold 0.80

    # Tag a run to avoid overwriting previous results:
    python evaluate.py booknlp \\
        --data-path ../../data/intermediate/BookNLP_output \\
        --run-label v2

Results are written to:
    ../../data/results/<pipeline_name>/evaluation_results.txt   (summary)
    ../../data/results/<pipeline_name>/per_book/<book>.json      (per-book detail)

Or, with --run-label:
    ../../data/results/<pipeline_name>_<run_label>/...

Book names expected by pipelines (must match XML file stems):
    emmelinetheorphan, fitzgerald, litigators, prideandprejudice, roomwithaview, thespy
"""

import os
import argparse

from loaders import REGISTRY
from core import run_evaluation

HERE                 = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLD_DIR     = os.path.join(HERE, "..", "..", "data", "raw", "quote_extraction_annotated")
DEFAULT_RESULTS_ROOT = os.path.join(HERE, "..", "..", "data", "results")
DEFAULT_THRESHOLD    = 0.90


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate dialogue extraction against Dialogism gold annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    ap.add_argument(
        "pipeline",
        choices=sorted(REGISTRY.keys()),
        help="Which extraction pipeline produced the predictions. "
             f"Available: {', '.join(sorted(REGISTRY.keys()))}",
    )
    ap.add_argument(
        "--data-path",
        required=True,
        help="Path to the pipeline's output. For booknlp, "
               "this is the root directory of per-book output files.",
    )
    ap.add_argument(
        "--gold-dir",
        default=DEFAULT_GOLD_DIR,
        help="Directory containing Dialogism <book>_annotated*.xml gold files. "
             f"Default: {DEFAULT_GOLD_DIR}",
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
        help="Optional tag appended to the results folder name. "
             "e.g. --run-label v2 → results/booknlp_v2/",
    )
    ap.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Parent directory for results. A subdirectory per pipeline "
             f"is created inside it. Default: {DEFAULT_RESULTS_ROOT}",
    )

    args = ap.parse_args()

    loader = REGISTRY[args.pipeline]

    results_name = args.pipeline
    if args.run_label:
        results_name += "_" + args.run_label
    results_dir = os.path.join(args.results_root, results_name)

    print(f"Loading predictions from: {args.data_path}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Gold XML directory: {args.gold_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Results will be written to: {results_dir}")
    print()

    # Pass gold_dir as ann_dir — loaders use it only for cross-reference warnings.
    # "No gold file" warnings from loaders are expected (no .ann files here).
    predictions = loader.load_predictions(args.data_path, args.gold_dir)

    if not predictions:
        print("ERROR: loader returned no predictions. Check --data-path.")
        return

    print(f"Loaded predictions for {len(predictions)} book(s).\n")

    run_evaluation(
        predictions=predictions,
        gold_dir=args.gold_dir,
        results_dir=results_dir,
        threshold=args.threshold,
        pipeline_name=args.pipeline,
    )


if __name__ == "__main__":
    main()
