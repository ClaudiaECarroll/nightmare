"""
Loader for QuoteAnnotator dialogue predictions.

QuoteAnnotator writes one JSON file per book:

    <data_path>/
        <book_id>_sampled.txt.json

Each JSON has a top-level "quotes" list.  Every element is an object like:

    {
      "id": 0,
      "text": "\"Oh dear!\"",
      "beginIndex": 713,
      ...
      "speaker": "Unknown",
      "canonicalSpeaker": "Unknown"
    }

We read the "text" field from every quote and return the raw strings.

Book-name mapping
-----------------
File names look like "105_persuasion_sampled.txt.json".  We strip
"_sampled.txt.json" to get "105_persuasion", which matches the gold
file "105_persuasion_brat.ann".
"""

import os
import json


def load_predictions(data_path, ann_dir):
    """Load QuoteAnnotator predictions from a directory of per-book JSON files.

    Parameters
    ----------
    data_path : str
        Directory containing <book>_sampled.txt.json files.
    ann_dir : str
        Path to gold .ann files (used only for missing-file warnings).

    Returns
    -------
    dict[str, list[str]]
        {book_name: [raw_quote_string, ...]}
        book_name has "_sampled" stripped so it matches gold .ann naming.
    """
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"QuoteAnnotator output directory not found: {data_path}"
        )

    predictions = {}

    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith(".txt.json"):
            continue

        # "105_persuasion_sampled.txt.json" -> "105_persuasion_sampled"
        stem = filename[: -len(".txt.json")]
        # "105_persuasion_sampled" -> "105_persuasion"
        book_name = stem[: -len("_sampled")] if stem.endswith("_sampled") else stem

        json_path = os.path.join(data_path, filename)
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        quotes = doc.get("quotes", [])
        strings = [q["text"] for q in quotes if q.get("text", "").strip()]

        predictions[book_name] = strings

        expected_ann = os.path.join(ann_dir, book_name + "_brat.ann")
        if not os.path.exists(expected_ann):
            print(f"[quoteannotator] Note: no gold file for '{book_name}' "
                  f"(looked for {expected_ann})")

    return predictions
