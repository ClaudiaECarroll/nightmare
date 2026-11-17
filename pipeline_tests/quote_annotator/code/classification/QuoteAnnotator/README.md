# QuoteAnnotator

## Run

From the project root, set `CORENLP_DIR` to the directory containing Stanford CoreNLP jars:

```bash
CORENLP_DIR=/path/to/corenlp bash code/classification/QuoteAnnotator/run_dialogism.sh
```

The script reads `data/raw/texts/*.txt`, builds a temporary file list, and writes JSON predictions to `data/intermediate/QuoteAnnotator_output/`.

## Requirements

- Java 8 or newer
- Stanford CoreNLP jars, available locally through `CORENLP_DIR`
