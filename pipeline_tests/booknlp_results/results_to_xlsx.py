"""
Convert each project's evaluation_results.txt into a per-book xlsx table.

Reads the "Per-Novel Results" table from
    <project>/data/results/booknlp/evaluation_results.txt
and writes booknlp_<name>.xlsx (one sheet, columns Book, Pred, Gold, TP, FP,
FN, Prec, Rec, F1) into this folder.

Usage (from pipeline_tests/):
    pip install openpyxl
    python booknlp_results/results_to_xlsx.py
"""

import os
import sys
from openpyxl import Workbook

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# project folder -> output name
PROJECTS = {
    "LitBank": "litbank",
    "RiQuA": "riqua",
    "quote_annotator": "quoteannotator",
    "nightmare": "nightmare",
}

HEADER = ["Book", "Pred", "Gold", "TP", "FP", "FN", "Prec", "Rec", "F1"]


def parse_per_book(path):
    rows = []
    in_table = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("=== Per-Novel Results"):
                in_table = True
                continue
            if not in_table:
                continue
            if line.startswith("=== "):
                break
            parts = line.split()
            if len(parts) != 9 or set(line.strip()) == {"-"} or parts[0] == "Book":
                continue
            book, counts, scores = parts[0], parts[1:6], parts[6:]
            rows.append([book] + [int(c) for c in counts] + [float(s) for s in scores])
    return rows


def main():
    for project, name in PROJECTS.items():
        src = os.path.join(ROOT, project, "data", "results", "booknlp", "evaluation_results.txt")
        if not os.path.exists(src):
            print(f"Skipping {project}: {src} not found", file=sys.stderr)
            continue
        rows = parse_per_book(src)
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(HEADER)
        for r in rows:
            ws.append(r)
        out = os.path.join(HERE, f"booknlp_{name}.xlsx")
        wb.save(out)
        print(f"{project}: wrote {len(rows)} books -> {out}")


if __name__ == "__main__":
    main()
