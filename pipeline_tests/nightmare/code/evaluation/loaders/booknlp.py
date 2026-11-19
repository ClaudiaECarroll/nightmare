"""Load BookNLP quote TSVs from one directory per sample."""

import csv
from pathlib import Path


def load_predictions(data_path):
    predictions = {}
    for book_dir in sorted(Path(data_path).iterdir()):
        if not book_dir.is_dir():
            continue
        book = book_dir.name
        candidates = [
            book_dir / (book + ".quotes"),
            book_dir / (book + ".txt.quotes"),
        ]
        files = [path for path in candidates if path.is_file()]
        if len(files) != 1:
            raise ValueError(f"Expected exactly one quotes file for {book}: {files}")
        with files[0].open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
            if not reader.fieldnames or "quote" not in reader.fieldnames:
                raise ValueError(f"Missing quote column in {files[0]}")
            predictions[book] = [
                row["quote"] for row in reader if row.get("quote", "").strip()
            ]
    return predictions
