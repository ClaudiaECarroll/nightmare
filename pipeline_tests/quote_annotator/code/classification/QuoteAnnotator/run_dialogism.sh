#!/usr/bin/env bash
# Run CoreNLP QuoteAnnotator on Dialogism plain-text novels.
#
# Input:  ../../data/raw/texts/*.txt
# Output: ../../data/intermediate/QuoteAnnotator_output/<book>.txt.json
#
# Run from the QuoteAnnotator/ directory or anywhere — uses absolute paths.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXTS_DIR="$SCRIPT_DIR/../../../data/raw/texts"
OUTPUT_DIR="$SCRIPT_DIR/../../../data/intermediate/QuoteAnnotator_output"
CORENLP_DIR="${CORENLP_DIR:-}"
FILELIST="$(mktemp "${TMPDIR:-/tmp}/dialogism-filelist.XXXXXX")"
trap 'rm -f "$FILELIST"' EXIT

if [[ -z "$CORENLP_DIR" || ! -d "$CORENLP_DIR" ]]; then
    echo "Set CORENLP_DIR to the directory containing the Stanford CoreNLP jars." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$TEXTS_DIR" ]]; then
    echo "Text input directory not found: $TEXTS_DIR" >&2
    exit 1
fi

# Build filelist of absolute paths to .txt files
find "$TEXTS_DIR" -type f -name '*.txt' -print | sort > "$FILELIST"
if [[ ! -s "$FILELIST" ]]; then
    echo "No input text files found in $TEXTS_DIR" >&2
    exit 1
fi
echo "Processing files:"
cat "$FILELIST"
echo ""

java -cp "$CORENLP_DIR/*" \
    -Xmx4g \
    edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,ssplit,quote \
    -filelist "$FILELIST" \
    -outputDirectory "$OUTPUT_DIR" \
    -outputFormat json \
    -quote.attributeQuotes false

echo ""
echo "Done. Output written to: $OUTPUT_DIR"
