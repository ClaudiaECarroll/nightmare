# Dialogism Dialogue Extraction Evaluation

Evaluates dialogue-extraction pipelines against the **Dialogism** corpus: 6
full novels annotated in a custom XML schema, where dialogue is marked as
`<quote speaker="...">...</quote>` (with `speaker="NONE"` for quoted-but-not-
dialogue text, e.g. headlines) and character coreference is marked with
nested `<mention>` tags.

Books: `emmelinetheorphan`, `fitzgerald`, `litigators`, `prideandprejudice`,
`roomwithaview`, `thespy`.

## Data flow

```
data/raw/quote_extraction_annotated/<book>_annotated[_newlinesreduced].xml
        │   gold XML — quote/speaker/mention spans
        │
        │  code/pre-processing/extract_plain_text.py
        │   (strips all markup, keeps whitespace/content intact)
        ▼
data/raw/texts/<book>.txt
        │
        │  each pipeline, run independently (see Pipelines below)
        ▼
data/intermediate/{BookNLP,QuoteAnnotator}_output/
        │
        │  code/evaluation/evaluate.py <pipeline> --data-path ...
        ▼
data/results/<pipeline>/
    evaluation_results.txt
    per_book/<id>.json
```

Gold is read directly from the XML files (not from a converted `.ann`
format) — see `load_gold_strings_from_xml()` in `code/evaluation/core.py`.

## Pipelines and results

| Pipeline | Driver | Precision | Recall | F1 |
|---|---|---|---|---|
| BookNLP | `code/classification/BookNLP/run_dialogism.py` — Python `booknlp` package | 0.739 | 0.739 | 0.739 |
| QuoteAnnotator | `code/classification/QuoteAnnotator/run_dialogism.sh` — Stanford CoreNLP `quote` annotator | 0.966 | 0.738 | 0.837 |

(13,414 gold quotes total across the 6 books; threshold 0.90, RapidFuzz
character-level matching.)

GutenTag was evaluated previously (precision 0.635, recall 0.487, F1 0.551),
but its driver lives outside this project and is not registered by the current
evaluator. Its existing prediction and result snapshots are retained.

**Historical note:** three additional pipelines — a regex extractor
("Claudia"), an R script ("DearReader"), and a second regex/notebook
pipeline ("Muqing") — were also built and evaluated against this corpus at
an earlier stage of the project (F1 0.791 / 0.766 / 0.740 respectively).
That code, its intermediate output, and its results live in an older,
pre-reorg sibling directory and were deliberately not migrated into this
folder — `code/evaluation/loaders/__init__.py` registers only the two supported
pipelines above.

## Running an evaluation

```bash
cd code/evaluation
python evaluate.py booknlp        --data-path ../../data/intermediate/BookNLP_output
python evaluate.py quoteannotator --data-path ../../data/intermediate/QuoteAnnotator_output
```
`--gold-dir` defaults to `data/raw/quote_extraction_annotated`, so it can be
omitted. Loaders will print "no gold file for `<book>`" warnings — these are
expected and harmless, since gold is loaded from the XML directly, not
through the loaders. See `code/README.md` for the full CLI reference.

Re-running a pipeline itself:
- `python code/pre-processing/extract_plain_text.py` (regenerate `data/raw/texts/*.txt` from the gold XML)
- `python code/classification/BookNLP/run_dialogism.py`
- `bash code/classification/QuoteAnnotator/run_dialogism.sh` (builds a temporary file list on each run)

## Adding a new pipeline

1. Run the pipeline on `data/raw/texts/*.txt`, writing output under `data/intermediate/`.
2. Copy `code/evaluation/loaders/_template.py` to a new loader module and implement `load_predictions()`.
3. Register it in `code/evaluation/loaders/__init__.py`.
4. Run `evaluate.py <your_pipeline_name> ...`.
