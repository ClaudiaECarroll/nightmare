# RiQua Dialogue Extraction Evaluation

Evaluates dialogue extraction against RiQua gold annotations using character-level fuzzy matching. RiQua's source texts are already natural prose, so no reconstruction step is required.

## Layout

```text
classification/BookNLP/run_riqua.py
classification/QuoteAnnotator/run_riqua.sh
evaluation/evaluate.py
evaluation/core.py
evaluation/loaders/{booknlp,quoteannotator}.py
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
python evaluate.py quoteannotator --data-path ../../data/intermediate/QuoteAnnotator_output
```

Gold annotations are read from `data/raw/labels/` by default. BookNLP requires the Python `booknlp` package and downloads its model on first run. To rerun QuoteAnnotator, set `CORENLP_DIR` to the directory containing Stanford CoreNLP jars, then run `bash code/classification/QuoteAnnotator/run_riqua.sh`.

## Scoring

Predicted and gold strings are normalised by removing whitespace and quotation marks, then compared with RapidFuzz's character-level ratio. A prediction is correct if its best similarity to any gold quote in that document is at least the threshold (default `0.90`).

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = harmonic mean of precision and recall
