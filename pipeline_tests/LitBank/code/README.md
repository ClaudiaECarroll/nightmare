# LitBank Dialogue Extraction Evaluation

Evaluates dialogue extraction against LitBank gold annotations using character-level fuzzy matching.

## Layout

```text
pre-processing/replicate_samples.py
classification/BookNLP/java/book-nlp/
classification/QuoteAnnotator/run_litbank.sh
classification/QuoteAnnotator/README.md
evaluation/evaluate.py
evaluation/core.py
evaluation/loaders/{booknlp,quoteannotator}.py
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
python evaluate.py quoteannotator --data-path ../../data/intermediate/QuoteAnnotator_output --ann-dir ../../data/raw/labels
```

To rerun QuoteAnnotator, first run `python code/pre-processing/replicate_samples.py`, set `CORENLP_DIR` to the Stanford CoreNLP installation, then run `bash code/classification/QuoteAnnotator/run_litbank.sh`. The script builds a temporary file list from reconstructed samples.

BookNLP uses the vendored Java implementation. Its README documents the external Stanford CoreNLP models jar requirement. The Python requirements file covers evaluation only.

## Scoring

Predicted and gold strings are normalised by removing whitespace and quotation marks, then compared with `difflib.SequenceMatcher`. A prediction is correct if its best similarity to any gold string in the same book is at least the threshold (default `0.90`).

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = harmonic mean of precision and recall
