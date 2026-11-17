#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEXTS_DIR="$PROJECT_ROOT/data/intermediate/reconstructed_samples"
OUTPUT_DIR="$PROJECT_ROOT/data/intermediate/QuoteAnnotator_output"
CORENLP_DIR="${CORENLP_DIR:-}"

if [[ -z "$CORENLP_DIR" || ! -d "$CORENLP_DIR" ]]; then
    echo "Set CORENLP_DIR to the directory containing the Stanford CoreNLP jars." >&2
    exit 1
fi

FILELIST="$(mktemp "${TMPDIR:-/tmp}/litbank-corenlp-filelist.XXXXXX")"
trap 'rm -f "$FILELIST"' EXIT

mkdir -p "$OUTPUT_DIR"
find "$TEXTS_DIR" -type f -name '*.txt' -print | sort > "$FILELIST"
if [[ ! -s "$FILELIST" ]]; then
    echo "No reconstructed text files found in $TEXTS_DIR; run replicate_samples.py first." >&2
    exit 1
fi

java -cp "$CORENLP_DIR/*" \
    -Xmx4g \
    edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,ssplit,quote \
    -filelist "$FILELIST" \
    -outputDirectory "$OUTPUT_DIR" \
    -outputFormat json \
    -quote.attributeQuotes false