#!/usr/bin/env python3
"""Run the notebook's custom Python BookNLP configuration on Nightmare samples."""

import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
BENCHMARK = Path(__file__).resolve().parents[5] / "benchmark"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BENCHMARK / "manual_samples")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "data/intermediate/BookNLP_output")
    parser.add_argument("--model-dir", type=Path, default=Path.home() / "booknlp_models")
    parser.add_argument("--bert-model-path", type=Path)
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.txt"))
    if not input_files:
        raise ValueError(f"No sample texts found in {args.input_dir}")
    params = {
        "pipeline": "entity,quote",
        "model": "custom",
        "entity_model_path": str(args.model_dir / "entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model"),
        "coref_model_path": str(args.model_dir / "coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model"),
        "quote_attribution_model_path": str(args.model_dir / "speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model"),
        "bert_model_path": str(args.bert_model_path or Path.home() / ".cache/huggingface/hub"),
    }
    for key in ("entity_model_path", "coref_model_path", "quote_attribution_model_path"):
        if not Path(params[key]).is_file():
            raise FileNotFoundError(f"Missing {key}: {params[key]}")
    if args.bert_model_path is not None and not args.bert_model_path.is_dir():
        raise FileNotFoundError(f"Missing BERT model directory: {args.bert_model_path}")
    existing = [str(args.output_dir / file.stem) for file in input_files
                if (args.output_dir / file.stem).exists()]
    if existing:
        raise FileExistsError(
            "Output directories already exist; choose a fresh --output-dir: "
            + ", ".join(existing)
        )

    import torch
    from booknlp.booknlp import BookNLP

    original_load = torch.load

    def load_without_position_ids(*load_args, **load_kwargs):
        state = original_load(*load_args, **load_kwargs)
        if isinstance(state, dict):
            state.pop("bert.embeddings.position_ids", None)
        return state

    torch.load = load_without_position_ids
    try:
        model = BookNLP("en", params)
        for input_file in input_files:
            book = input_file.stem
            output_dir = args.output_dir / book
            output_dir.mkdir(parents=True)
            print(f"Processing {book}", flush=True)
            model.process(str(input_file), str(output_dir), book)
            if not (output_dir / (book + ".quotes")).is_file():
                raise RuntimeError(f"BookNLP did not produce a quotes file for {book}")
    finally:
        torch.load = original_load


if __name__ == "__main__":
    main()
