#!/usr/bin/env python3
"""
Extract plain text from Dialogism annotated XML files.

Strips <quote> tags (and the XML declaration / <doc> wrapper) to produce
clean .txt files that pipelines like BookNLP can process.

Tag content is preserved; only the markup is removed. Whitespace and paragraph
structure are kept intact so that character-offset-based tools work correctly.

Usage:
    python extract_plain_text.py

Reads:  ../../data/raw/quote_extraction_annotated/*.xml
Writes: ../../data/raw/texts/<book_name>.txt

Book name mapping (strips _annotated[_newlinesreduced] suffix):
    emmelinetheorphan_annotated_newlinesreduced.xml -> emmelinetheorphan.txt
    fitzgerald_annotated.xml                        -> fitzgerald.txt
    litigators_annotated.xml                        -> litigators.txt
    prideandprejudice_annotated.xml                 -> prideandprejudice.txt
    roomwithaview_annotated.xml                     -> roomwithaview.txt
    thespy_annotated.xml                             -> thespy.txt
"""

import os
import re
import glob

HERE    = os.path.dirname(os.path.abspath(__file__))
XML_DIR = os.path.join(HERE, "..", "..", "data", "raw", "quote_extraction_annotated")
OUT_DIR = os.path.join(HERE, "..", "..", "data", "raw", "texts")


def xml_stem_to_book_name(xml_filename):
    """Strip _annotated[_newlinesreduced] suffix from an XML filename stem."""
    stem = xml_filename.replace(".xml", "")
    for suffix in ("_annotated_newlinesreduced", "_annotated"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def extract_plain_text(xml_path):
    """Return the plain text of an annotated XML file with all tags stripped.

    Reads the file as raw text and removes XML markup with regex, preserving
    all whitespace and literary content exactly as it appears.
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = re.sub(r"<\?xml[^?]*\?>", "", raw)   # XML declaration
    raw = re.sub(r"</?doc>", "", raw)           # <doc> wrapper
    raw = re.sub(r"<characters>.*?</characters>", "", raw, flags=re.DOTALL)  # character/alias metadata block
    raw = re.sub(r"</?text>", "", raw)          # <text> wrapper (present in some files)
    raw = re.sub(r"<quote[^>]*>", "", raw)      # opening <quote ...> tags
    raw = re.sub(r"</quote>", "", raw)          # closing </quote> tags
    raw = re.sub(r"<mention[^>]*>", "", raw)    # opening <mention ...> coref tags (present in some files)
    raw = re.sub(r"</mention>", "", raw)        # closing </mention> tags
    raw = raw.replace("&quot;", '"')            # decode literal quotation marks (present in some files)

    return raw


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    xml_files = sorted(glob.glob(os.path.join(XML_DIR, "*.xml")))
    if not xml_files:
        print(f"No XML files found in {XML_DIR}")
        return

    for xml_path in xml_files:
        xml_filename = os.path.basename(xml_path)
        book_name    = xml_stem_to_book_name(xml_filename)
        out_path     = os.path.join(OUT_DIR, book_name + ".txt")

        text = extract_plain_text(xml_path)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  {xml_filename}  ->  {book_name}.txt  ({len(text):,} chars)")

    print(f"\nDone. Plain text files written to:\n  {os.path.realpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
