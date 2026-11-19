# LitBank Dialogue Extraction Evaluation

Evaluates dialogue extraction against LitBank gold annotations using character-level fuzzy matching.

## Layout

```text
pre-processing/replicate_samples.py
classification/BookNLP/run_litbank.py
evaluation/evaluate.py
evaluation/core.py
evaluation/loaders/booknlp.py
```

The evaluator reads predictions from `data/intermediate/`, gold annotations from `data/raw/labels/`, and writes reports under `data/results/`.

## Dependencies and use

From the repository root, install the evaluation dependencies:

```bash
python -m pip install -r requirements.txt
```

From `code/evaluation/`, evaluate saved predictions:

```bash
python evaluate.py booknlp --data-path ../../data/intermediate/BookNLP_output --ann-dir ../../data/raw/labels
```

To rebuild the BookNLP input texts, run `python code/pre-processing/replicate_samples.py`; it writes `data/intermediate/reconstructed_samples/`.

To rerun BookNLP, run `python code/pre-processing/replicate_samples.py`, then `python code/classification/BookNLP/run_litbank.py`. BookNLP requires the Python `booknlp` package and downloads its model on first run. The Python requirements file covers evaluation only.

## Scoring

Predicted and gold strings are normalised by removing whitespace and quotation marks, then compared with RapidFuzz's character-level ratio. A prediction is correct if its best similarity to any gold string in the same book is at least the threshold (default `0.90`).

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = harmonic mean of precision and recall
