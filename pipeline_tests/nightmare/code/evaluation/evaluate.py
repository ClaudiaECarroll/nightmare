#!/usr/bin/env python3
"""Evaluate pipeline output against the Nightmare benchmark."""

import argparse
from pathlib import Path

from core import load_gold_strings, run_evaluation
from loaders import REGISTRY

PROJECT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(__file__).resolve().parents[4] / "benchmark"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", choices=sorted(REGISTRY))
    parser.add_argument("--data-path", type=Path,
                        default=PROJECT / "data/intermediate/BookNLP_output")
    parser.add_argument("--gold-file", type=Path, default=BENCHMARK / "final_benchmark.tsv")
    parser.add_argument("--samples-dir", type=Path, default=BENCHMARK / "manual_samples")
    parser.add_argument("--results-root", type=Path, default=PROJECT / "data/results")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--run-label", help="Append a label to the results directory")
    args = parser.parse_args()
    if args.run_label and (Path(args.run_label).name != args.run_label or args.run_label in (".", "..")):
        parser.error("--run-label must be a single directory name")
    name = args.pipeline + ("_" + args.run_label if args.run_label else "")
    gold = load_gold_strings(args.gold_file, args.samples_dir)
    predictions = REGISTRY[args.pipeline].load_predictions(args.data_path)
    run_evaluation(predictions, gold, args.results_root / name, args.threshold, args.pipeline)


if __name__ == "__main__":
    main()
