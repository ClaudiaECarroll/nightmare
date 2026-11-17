"""
Loader for BookNLP dialogue predictions.

BookNLP writes one output directory per book inside a root directory:

    <data_path>/
        <book_id>/
            <book_id>.quotes     <- tab-separated, has a `quote` column

Each .quotes file is a TSV where every row is one detected quote.  We read
the `quote` column and return the raw strings (un-normalised).

Book-name mapping
-----------------
BookNLP folder names often end with "_sampled" (e.g. "jane_eyre_sampled"),
but the gold .ann files use the base name ("jane_eyre_brat.ann").  We strip
the "_sampled" suffix so the returned dict keys match the gold filenames.
"""

import os
import csv


def load_predictions(data_path, ann_dir):
    """Load BookNLP predictions from a directory of per-book folders.

    Parameters
    ----------
    data_path : str
        Root directory containing one subfolder per book, each with a
        <folder_name>.quotes TSV file inside it.
    ann_dir : str
        Path to gold .ann files (unused for loading, but available if needed
        for cross-referencing).

    Returns
    -------
    dict[str, list[str]]
        {book_name: [raw_quote_string, ...]}
        book_name has "_sampled" stripped so it matches the gold .ann naming.
    """
    predictions = {}

    for folder in sorted(os.listdir(data_path)):
        book_dir = os.path.join(data_path, folder)
        if not os.path.isdir(book_dir):
            continue

        quotes_path = os.path.join(book_dir, folder + ".quotes")
        if not os.path.exists(quotes_path):
            print(f"[booknlp] Skipping {folder}: no .quotes file")
            continue

        # Strip "_sampled" so keys align with gold file names.
        # e.g. "jane_eyre_sampled" -> "jane_eyre"
        book_name = folder.replace("_sampled", "")

        strings = []
        with open(quotes_path, "r", encoding="utf-8") as f:
            # BookNLP .quotes files are tab-delimited with a header row.
            # quoting=QUOTE_NONE because the quote text itself contains
            # quotation-mark characters that would confuse the CSV parser.
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                text = row.get("quote", "") or ""
                if text.strip():
                    strings.append(text)

        predictions[book_name] = strings

    return predictions
