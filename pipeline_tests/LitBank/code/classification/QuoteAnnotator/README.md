# QuoteAnnotator

## Run

Generate reconstructed samples first, then set the CoreNLP installation and run:

```bash
python code/pre-processing/replicate_samples.py
CORENLP_DIR=/path/to/corenlp bash code/classification/QuoteAnnotator/run_litbank.sh
## Run

Generate reconstructed samples first, then set the CoreNLP installation and run:

```bash
python code/pre-processing/replicate_samples.py
CORENLP_DIR=/path/to/corenlp bash code/classification/QuoteAnnotator/run_litbank.sh
```

The runner builds a temporary file list from
`data/intermediate/reconstructed_samples/` and writes predictions to
`data/intermediate/QuoteAnnotator_output/`.

## Direct Java command

The runner builds a temporary file list from
`data/intermediate/reconstructed_samples/` and writes predictions to
`data/intermediate/QuoteAnnotator_output/`.

## Direct Java command
java -cp "/path/to/corenlp/*" -Xmx4g edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,ssplit,quote \
    -outputFormat json \
    -quote.attributeQuotes false

## Requirements
- you need to have Java above 8
- you need CoreNLP

