# LitBank Dialogue Extraction Evaluation

This project measures how accurately different dialogue/quote-extraction
pipelines find spoken dialogue in prose, using 100 novels from the
[LitBank](https://github.com/dbamman/litbank) corpus as gold-standard
ground truth.

It has two halves:

1. **Run pipelines** — each pipeline reads a novel and outputs the quotes
   it thinks are dialogue. Pipelines are set up and run independently of
   each other (see [Pipelines](#pipelines) below).
2. **Evaluate** — `code/evaluation/evaluate.py` scores any pipeline's
   output against LitBank's gold annotations using character-level fuzzy
   matching, and writes a precision/recall/F1 report.

These two halves are decoupled on purpose: a pipeline's output is just a
directory or file dropped into `data/intermediate/`, and the evaluator
doesn't care how it got there.

## Data flow

```
data/raw/texts/<id>.txt              raw Project Gutenberg text
data/raw/brat_texts/<id>_brat.txt     same text, brat-tokenized
data/raw/labels/<id>_brat.ann         LitBank gold annotations
        │
        │  code/pre-processing/replicate_samples.py
        ▼
data/intermediate/reconstructed_samples/<id>_sampled.txt
        │   "unprocessed" text reconstructed from the brat tokenization —
        │   this is the realistic input each pipeline actually runs on
        │
        │  each pipeline, run independently (see Pipelines below)
        ▼
data/intermediate/{BookNLP,QuoteAnnotator}_output/
        │   one pipeline's raw predictions, one file/folder per book
        │
        │  code/evaluation/evaluate.py <pipeline> --data-path ... --ann-dir data/raw/labels
        ▼
data/results/<pipeline>/
    evaluation_results.txt          precision / recall / F1 summary
    per_book/<id>.json              per-quote detail (why each prediction passed/failed)
```

100 books go in; **14 have a 0-byte gold `.ann` file** (empty gold — LitBank
never annotated dialogue for these). Any prediction on those books is a
forced false positive with no chance of a true positive. The evaluation
report breaks out totals with and without these 14 books so they don't
quietly depress every pipeline's precision.

## Gold annotation format

`data/raw/labels/<id>_brat.ann` is tab-separated. The two line types that
matter for this project:

```
QUOTE   <id>   <start_sent>   <start_tok>   <end_sent>   <end_tok>   <tokenized quote text>
ATTRIB  ...                                                          (speaker attribution — not used by evaluate.py)
```

`evaluate.py` only reads `QUOTE` lines (the 7th tab-separated field is the
gold dialogue string); `ATTRIB` lines are skipped. Not every book has
`ATTRIB` lines even when it has `QUOTE` lines.

## Pipelines

| Pipeline | Status | Driver location | Notes |
|---|---|---|---|
| BookNLP | done | `code/classification/BookNLP/java/book-nlp/` (vendored 2014 Java BookNLP — see its own README for how to run it) | Needs a Stanford CoreNLP models jar dropped into its `lib/` folder; run manually via `./runjava`. |
| QuoteAnnotator | done | `code/classification/QuoteAnnotator/run_litbank.sh` | Requires Java and Stanford CoreNLP; set `CORENLP_DIR` before running. |
| Claudia / DearReader / Muqing | not built | — | Planned pipelines (regex script, R script, notebook) that were never implemented. No loader exists for them either. |

GutenTag was evaluated previously (precision 0.968, recall 0.771, F1 0.858),
but its driver is outside this project and is not registered by the evaluator.
Its existing prediction and result snapshots are retained.

Whichever pipeline you run, its output has to land in
`data/intermediate/<Pipeline>_output/` in the naming convention its loader
expects (see `code/evaluation/loaders/<pipeline>.py` docstrings for the
exact filename patterns each one reads).

## Running an evaluation

```bash
cd code/evaluation
python evaluate.py booknlp \
    --data-path ../../data/intermediate/BookNLP_output \
    --ann-dir   ../../data/raw/labels

python evaluate.py quoteannotator \
    --data-path ../../data/intermediate/QuoteAnnotator_output \
    --ann-dir   ../../data/raw/labels

```

`--ann-dir` defaults to `data/raw/labels` already, so it can be omitted. To
rerun QuoteAnnotator, first run `replicate_samples.py`, then use the script in
`code/classification/QuoteAnnotator/` with `CORENLP_DIR` set.
See `code/README.md` for the full CLI reference (thresholds, run labels,
adding a new pipeline's loader) and how the fuzzy-matching score itself
works.

## Adding a new pipeline

1. Run the pipeline on `data/intermediate/reconstructed_samples/*.txt`,
   writing its output somewhere under `data/intermediate/`.
2. Copy `code/evaluation/loaders/_template.py` to a new loader module and
   implement `load_predictions()` for that output format.
3. Register it in `code/evaluation/loaders/__init__.py`.
4. Run `evaluate.py <your_pipeline_name> ...` as above.

Full details in `code/README.md`.
