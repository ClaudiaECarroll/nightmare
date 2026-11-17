# RiQua Dialogue Extraction Evaluation

Evaluates dialogue-extraction pipelines against the **RiQuA (Rich Quotation
Annotation Corpus)**, v0.1, from the Institute for Natural Language
Processing, University of Stuttgart.

RiQuA annotates 15 literary excerpts for direct/indirect quotations, cues,
speakers, and addressees, in BRAT standoff format. The document set matches
the Columbia Quoted Speech Attribution Corpus (Elson & McKeown 2010):
Austen's *Emma* (×3 passages), Chekhov (×3 stories), Dickens' *A Christmas
Carol*, four Conan Doyle stories, Flaubert's *Madame Bovary* (×2 passages),
and Twain's *The Adventures of Tom Sawyer* (×2 passages). Two independent
annotators plus a merged/adjudicated gold set are provided — see
`data/raw/README` for the full corpus documentation. This harness evaluates
against the merged annotations, per the corpus authors' own recommendation.

Unlike LitBank, RiQua's raw texts are already natural running prose, so
there's no pre-processing/reconstruction step here.

## Data flow

```
data/raw/
    annotator-1/, annotator-2/, merged/   independent + adjudicated BRAT annotations
    labels/                               <book>_brat.ann — copy of merged/*.ann
    texts/                                <book>.txt — copy of merged/*.txt
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

(`labels/` and `texts/` are exact copies of `merged/`, just renamed to match
the evaluator's expected file naming — not a separate data source.)

## Pipelines and results

| Pipeline | Driver | Precision | Recall | F1 |
|---|---|---|---|---|
| BookNLP | `code/classification/BookNLP/run_riqua.py` — modern **Python** `booknlp` package (not the vendored Java jar LitBank uses) | 0.856 | 0.834 | 0.845 |
| QuoteAnnotator | `code/classification/QuoteAnnotator/run_riqua.sh` — Stanford CoreNLP `quote` annotator | 0.919 | 0.911 | 0.915 |

(5,963 gold quotes total across the 15 books; threshold 0.90, RapidFuzz character-level matching — see `code/evaluation/core.py`.)

GutenTag was evaluated previously (precision 0.968, recall 0.771, F1 0.858),
but its driver lives outside this project and is not registered by the current
evaluator. Its existing prediction and result snapshots are retained.

## Running an evaluation

```bash
cd code/evaluation
python evaluate.py booknlp        --data-path ../../data/intermediate/BookNLP_output
python evaluate.py quoteannotator --data-path ../../data/intermediate/QuoteAnnotator_output
```
`--ann-dir` defaults to `data/raw/labels`, so it can be omitted. See
`code/README.md` for the full CLI reference.

Re-running a pipeline itself:
- `python code/classification/BookNLP/run_riqua.py`
- `bash code/classification/QuoteAnnotator/run_riqua.sh` (builds a temporary file list on each run)

## Adding a new pipeline

1. Run the pipeline on `data/raw/texts/*.txt`, writing output under `data/intermediate/`.
2. Copy `code/evaluation/loaders/_template.py` to a new loader module and implement `load_predictions()`.
3. Register it in `code/evaluation/loaders/__init__.py`.
4. Run `evaluate.py <your_pipeline_name> ...`.
