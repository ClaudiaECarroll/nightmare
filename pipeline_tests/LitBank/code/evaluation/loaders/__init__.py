"""
Loader registry.

Each loader module in this package exposes one function:

    load_predictions(data_path: str, ann_dir: str) -> dict[str, list[str]]

It returns {book_name: [raw_dialogue_string, ...]} where book_name
corresponds to a gold file at <ann_dir>/<book_name>_brat.ann.

The ann_dir parameter is passed through so that loaders which discover
books by scanning a directory (like booknlp) can optionally cross-reference
against available gold files, while loaders that read from a single CSV
(like authentic_regex) can ignore it.

Raw strings are returned UN-normalised — the evaluation core handles
normalisation uniformly so every pipeline is treated the same way.

To add a new pipeline:
  1. Create a new .py file in this directory with a load_predictions() function.
  2. Import it below and add an entry to REGISTRY.
"""

from . import booknlp
from . import quoteannotator

# Maps the CLI pipeline name to its module.
# Each module must have: load_predictions(data_path, ann_dir) -> dict[str, list[str]]
REGISTRY = {
    "booknlp": booknlp,
    "quoteannotator": quoteannotator,
}
