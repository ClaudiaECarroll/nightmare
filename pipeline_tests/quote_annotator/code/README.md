# Dialogue Extraction Evaluation

Evaluates dialogue extraction against Dialogism gold XML annotations using character-level fuzzy matching.

## Layout

```text
pre-processing/extract_plain_text.py
classification/BookNLP/run_dialogism.py
evaluation/evaluate.py
evaluation/core.py
evaluation/loaders/booknlp.py
```

Reports are written under `data/results/` by default.

## Dependencies and use

From the repository root, install the evaluation dependencies:

```bash
python -m pip install -r requirements.txt
```

From `code/evaluation/`, evaluate saved predictions:

```bash
python evaluate.py booknlp --data-path ../../data/intermediate/BookNLP_output
```

Gold annotations are read directly from XML through `--gold-dir`; the default is `data/raw/quote_extraction_annotated`.

To regenerate plain text, run `python code/pre-processing/extract_plain_text.py` from the repository root. BookNLP requires the Python `booknlp` package and downloads its model on first run.

## Scoring

Predicted and gold strings are normalised by removing whitespace and quotation marks, then compared with RapidFuzz's character-level ratio. A prediction is correct if its best similarity to any gold quote in that book is at least the threshold (default `0.90`).

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = harmonic mean of precision and recall
