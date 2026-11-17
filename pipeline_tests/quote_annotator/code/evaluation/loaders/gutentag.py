"""
Loader for GutenTag dialogue predictions.

The driver script (GutenTag/run_litbank.py) writes one JSON file per book:

    <data_path>/
        <book_name>.json     e.g. 105_persuasion.json

Each JSON has the structure:
    {"quotes": ["quote text 1", "quote text 2", ...]}

where each string is the space-joined token sequence of one <said> span
as detected by GutenTag's SaidTagger.

Book-name mapping
-----------------
The JSON filenames already have "_sampled" stripped (that was done in the
driver script), so "105_persuasion.json" maps directly to the gold file
"105_persuasion_brat.ann".
"""

import os
import json


def load_predictions(data_path, ann_dir):
    """Load GutenTag predictions from a directory of per-book JSON files.

    Parameters
    ----------
    data_path : str
        Directory containing <book_name>.json files.
    ann_dir : str
        Path to gold .ann files (used only for missing-file warnings).

    Returns
    -------
    dict[str, list[str]]
        {book_name: [raw_quote_string, ...]}
    """
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"GutenTag output directory not found: {data_path}"
        )

    predictions = {}

    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith(".json"):
            continue

        book_name = filename[:-len(".json")]
        json_path = os.path.join(data_path, filename)

        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        quotes = doc.get("quotes", [])
        strings = [q for q in quotes if q.strip()]

        predictions[book_name] = strings

        expected_ann = os.path.join(ann_dir, book_name + "_brat.ann")
        if not os.path.exists(expected_ann):
            print(f"[gutentag] Note: no gold file for '{book_name}' "
                  f"(looked for {expected_ann})")

    return predictions
