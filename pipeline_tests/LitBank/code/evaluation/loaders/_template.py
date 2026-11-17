"""
TEMPLATE: Copy this file to create a loader for a new extraction pipeline.

Steps:
  1. Copy this file:  cp _template.py my_new_pipeline.py
  2. Implement load_predictions() below.
  3. In __init__.py, import your module and add it to REGISTRY.

Your load_predictions function must return:

    {book_name: [raw_dialogue_string, ...], ...}

Where:
  - book_name matches the gold .ann file: <ann_dir>/<book_name>_brat.ann
  - Each string is a raw, un-normalised dialogue utterance as extracted
    by your pipeline.  The evaluation core handles normalisation.
"""


def load_predictions(data_path, ann_dir):
    """Load dialogue predictions from <describe your format here>.

    Parameters
    ----------
    data_path : str
        Path to <describe what this points to for your pipeline>.
    ann_dir : str
        Path to the directory of gold .ann files.

    Returns
    -------
    dict[str, list[str]]
        {book_name: [raw_dialogue_string, ...]}
    """
    raise NotImplementedError(
        "This is a template. Copy it and implement load_predictions()."
    )
