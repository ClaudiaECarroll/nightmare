# Nightmare Dialogue Extraction Evaluation

This test runs Python BookNLP on the 12 samples in
[`../../benchmark/manual_samples/`](../../benchmark/manual_samples/) and evaluates
its dialogue predictions against `speech` rows in
[`../../benchmark/final_benchmark.tsv`](../../benchmark/final_benchmark.tsv).
The benchmark remains the source of truth; no copy is needed here.

## Layout

```text
code/classification/BookNLP/run_nightmare.py
code/evaluation/evaluate.py
code/evaluation/core.py
code/evaluation/loaders/booknlp.py
data/intermediate/BookNLP_output/          predictions used in the saved report
data/intermediate/legacy_BookNLP_output/   other historical output files
data/results/booknlp/                      saved evaluation and per-book details
```

The original notebook, `booknlp_on_nightmare.ipynb`, is retained for historical
context; its absolute paths refer to the old layout. Some quote files in the
legacy output differ from the files used for evaluation, so both are preserved.

## Setup

From this directory, install evaluator dependencies:

```bash
python -m pip install -r requirements.txt
```

To run BookNLP, use Python 3.10 and install the Python packages and spaCy model
used by the other tests:

```bash
python -m pip install booknlp torch "setuptools<81"
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

The historical Nightmare run used `entity,quote` and the weights for BookNLP's
`big` model, not the other tests' `small` configuration. Place these BookNLP
model files in `~/booknlp_models`, or specify their directory with `--model-dir`:

- `entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model`
- `coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model`
- `speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model`

BookNLP's `big` configuration downloads these weights into `~/booknlp_models`
on initialization. Alternatively, download them directly:

```bash
mkdir -p ~/booknlp_models
curl -fL https://people.ischool.berkeley.edu/~dbamman/booknlp_models/entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model -o ~/booknlp_models/entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model
curl -fL https://people.ischool.berkeley.edu/~dbamman/booknlp_models/coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model -o ~/booknlp_models/coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model
curl -fL https://people.ischool.berkeley.edu/~dbamman/booknlp_models/speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model -o ~/booknlp_models/speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model
```

The custom BERT path defaults to `~/.cache/huggingface/hub`; override it with
`--bert-model-path` if needed. Hugging Face may download additional model files
on first use.
The runner removes the historical `bert.embeddings.position_ids` key in memory
when loading weights, without modifying the model files.

## Run and evaluate

Commands below run from this directory. The saved predictions can be evaluated
without BookNLP:

```bash
python code/evaluation/evaluate.py booknlp
```

To avoid overwriting the saved predictions, run BookNLP into a new directory:

```bash
python code/classification/BookNLP/run_nightmare.py --output-dir data/intermediate/BookNLP_new
python code/evaluation/evaluate.py booknlp --data-path data/intermediate/BookNLP_new --run-label new
```

The runner refuses to overwrite an existing sample output directory. Both
commands offer `--help` for path overrides; the evaluator also accepts
`--threshold` and `--results-root`. A missing or unexpected sample is an error,
while a valid header-only `.quotes` file represents zero predictions.

The evaluator retains the notebook's whitespace/quotation-mark normalization,
RapidFuzz similarity threshold (default 0.90), and independent prediction and
gold matching. For compatibility, recall is computed as TP / (TP + FN), as in
the original report, even though several predictions can match one gold quote.
The saved report has precision 0.751, recall 0.785 and F1 0.767; sample 6124 has
zero predictions and 66 missed gold quotes. Re-running the pipeline may change
these results.
