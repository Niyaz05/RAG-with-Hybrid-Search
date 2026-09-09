"""
DAY 1 — STEP 1: PDF Parsing
============================
Goal: turn each PDF into a list of {source, week, slide, text} records —
one record per page/slide.

WHY PER-PAGE, NOT THE WHOLE DOCUMENT AT ONCE?
Your slide decks already segment content into topics, one per page. If we
concatenated the whole PDF into one giant string, we'd throw that structure
away and have to re-invent it later during chunking. So: respect the
boundary the document already gives us for free.

KEY LIBRARY CONCEPT — PyMuPDF (imported as `pymupdf`, formerly `fitz`):
A PDF page's `get_text()` reads whatever text layer is embedded in the
PDF. This ONLY works because your slides were exported from something
like PowerPoint (real, selectable text) — not scanned photographs of
paper. If you ever feed this a scanned PDF, get_text() will return
nothing or garbage, and you'd need OCR (e.g. Tesseract) instead. Always
sanity-check extracted text on a new document type before trusting it.
"""

from typing import Optional

import json
import re
from pathlib import Path

import pymupdf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Path(__file__).parent = the folder this script lives in (ingest/)
# .parent again = the project root
# This pattern makes the script runnable from ANY working directory,
# instead of hardcoding "../data" which breaks depending on where you `cd`.
PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"
OUT_PATH = Path(__file__).parent.parent / "data" / "parsed_pages.json"

# LEARNING NOTE — why we dropped the filename->label lookup table:
# We originally mapped filenames to hardcoded "Week 1/2/3/4" labels. Two
# problems with that, discovered the hard way:
#   1. It's fragile — the table and the filenames can drift out of sync
#      (exactly what happened: the dict got edited and every label quietly
#      became wrong, with no error to alert us).
#   2. It was ALSO factually wrong even when in sync — these four PDFs
#      aren't sequential "weeks 1-4" of a course. Look at the real dates:
#      Jan 5-9, Feb 2-6, March 16-20, April 6-10 — there are multi-week
#      gaps between them. Calling them "Week 1/2/3/4" invents a sequence
#      that doesn't exist.
#
# The fix: extract the ACTUAL date range from each PDF's own text (it's
# already sitting right there on slide 1 — "6th April. to 10th April. 2026")
# instead of guessing from the filename. This is a more general lesson:
# prefer deriving metadata from the document's own content over an
# external lookup table you have to keep manually in sync.

# Matches patterns like "6th April. to 10th April. 2026" / "05th Jan. to 09th Jan. 2026"
DATE_RANGE_PATTERN = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?\.?\s+([A-Za-z]+)\.?\s+to\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?\.?\s+([A-Za-z]+)\.?\s+(\d{4})"
)


# ---------------------------------------------------------------------------
# PURE FUNCTIONS (no file I/O — this is what makes them easy to unit test)
# ---------------------------------------------------------------------------
# LEARNING NOTE: notice these functions take plain strings in, return plain
# strings/values out — no reading files, no PyMuPDF objects. That's a
# deliberate design choice called "separating pure logic from I/O." It means
# test_parse_pdf.py can test THESE functions directly with zero PDF files
# on disk, which is exactly what you'll see in the test file.

def extract_week_label(first_page_text: str) -> Optional[str]:
    """
    Pull the real date range straight out of a PDF's own cover-slide text,
    e.g. "6th April. to 10th April. 2026" -> "6-10 April 2026".

    Returns None if the pattern isn't found, so callers can fall back to
    something else rather than crashing on an unexpected slide layout.
    """
    match = DATE_RANGE_PATTERN.search(first_page_text)
    if not match:
        return None

    start_day, start_month, end_day, _end_month, year = match.groups()
    # int() strips leading zeros ("06" -> 6) so labels are consistently
    # formatted regardless of how the source text zero-padded the day.
    # We only need one month name (start_month) since both ends of a
    # single week's range are always the same month in this course's decks.
    return f"{int(start_day)}-{int(end_day)} {start_month} {year}"


def week_label_for(filename: str) -> str:
    """
    Fallback label derived purely from the filename, used ONLY when
    extract_week_label() can't find a date range in the PDF's own text
    (e.g. a differently-formatted cover slide in a future PDF).
    """
    stem = filename.lower().replace(".pdf", "").replace("_", " ")
    return stem.title()


def clean_text(text: str) -> str:
    """
    PDF text extraction often leaves messy whitespace: multiple spaces
    where a table/column used to be, or long runs of blank lines between
    text blocks. We normalize that here so downstream chunking works on
    predictable text.

    re.sub(r"[ \t]+", " ", text)   -> collapse runs of spaces/tabs to one
    re.sub(r"\n{3,}", "\n\n", text) -> collapse 3+ blank lines to exactly 2
                                        (we KEEP double newlines on purpose —
                                        chunk.py uses "\n\n" as a paragraph
                                        boundary marker later)
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# I/O FUNCTIONS (these DO touch the filesystem / PyMuPDF)
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Open one PDF, walk every page, extract text, return one dict per page.

    pymupdf.open(path) gives you a `Document` object you can index like a
    list: doc[0] is page 1, doc[1] is page 2, etc. — this is why we use
    `page_index + 1` below to report 1-indexed "slide 7" instead of the
    programmer-friendly but human-confusing "slide 6".
    """
    records = []
    doc = pymupdf.open(pdf_path)

    # Try to derive the week label from the PDF's OWN cover-slide text
    # first (page 0, before any cleaning distorts spacing the regex relies
    # on). Only fall back to guessing from the filename if that fails.
    first_page_raw_text = doc[0].get_text() if len(doc) > 0 else ""
    week = extract_week_label(first_page_raw_text)
    if week is None:
        week = week_label_for(pdf_path.name)
        print(f"  (note: couldn't find a date range in {pdf_path.name}'s text, "
              f"falling back to filename-derived label: {week!r})")

    for page_index in range(len(doc)):
        page = doc[page_index]
        raw_text = page.get_text()          # <- the actual PDF text-layer read
        text = clean_text(raw_text)

        # Some slides (rare) extract to almost nothing — e.g. a slide that's
        # 95% an image with a 2-word caption. We skip those rather than
        # index a near-empty, useless chunk. Tune this threshold if you
        # notice real content being skipped.
        if len(text) < 5:
            continue

        records.append(
            {
                "source": pdf_path.name,
                "week": week,
                "slide": page_index + 1,
                "text": text,
            }
        )

    doc.close()  # always close PDF handles explicitly — don't rely on GC
    return records


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(
            f"No PDFs found in {PDF_DIR}.\n"
            f"Put jan_2026.pdf, feb_2026.pdf, march_2026.pdf, april_2026.pdf there first."
        )

    all_records = []
    for pdf_path in pdf_files:
        recs = parse_pdf(pdf_path)
        print(f"{pdf_path.name}: {len(recs)} slides parsed")
        all_records.extend(recs)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False keeps things like en-dashes/curly quotes readable
    # in the JSON file instead of escaped as \u2013 etc.
    OUT_PATH.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))

    print(f"\nTotal slides parsed: {len(all_records)}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()