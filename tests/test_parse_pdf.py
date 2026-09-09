"""
TESTS for ingest/parse_pdf.py
==============================
Run with:  pytest tests/test_parse_pdf.py -v

LEARNING NOTE — why two kinds of tests here?
1. UNIT tests: test one pure function in total isolation, with inputs you
   made up by hand. Fast, no dependencies, pinpoint exactly what broke.
2. INTEGRATION test: runs the real parse_pdf() against your real PDFs on
   disk. Slower, and SKIPPED automatically if you haven't put your PDFs
   in data/pdfs/ yet — so this test file works from day one, before you've
   even added your files, and gets stricter automatically once you have.
"""

import sys
from pathlib import Path

import pytest

# Make `ingest/` importable when running pytest from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

# pyrefly: ignore [missing-import]
from parse_pdf import clean_text, week_label_for, extract_week_label, parse_pdf, PDF_DIR


# ---------------------------------------------------------------------------
# UNIT TESTS — pure functions, no files touched
# ---------------------------------------------------------------------------

def test_extract_week_label_from_real_slide_text():
    """
    These are the EXACT four date-range phrasings from your real slide 1s
    (Jan/Feb/March/April). This test would have caught the original bug
    immediately, because it checks the actual source-of-truth text instead
    of a hand-maintained lookup table that can silently drift out of sync.
    """
    assert extract_week_label("... 6th April. to 10th April. 2026 ...") == "6-10 April 2026"
    assert extract_week_label("... 2nd Feb. to 06th Feb. 2026 ...") == "2-6 Feb 2026"
    assert extract_week_label("... 05th Jan. to 09th Jan. 2026 ...") == "5-9 Jan 2026"
    assert extract_week_label("... 16th March. to 20th March. 2026 ...") == "16-20 March 2026"


def test_extract_week_label_returns_none_when_pattern_absent():
    """
    A slide with no date-range phrase (e.g. a future PDF with a different
    cover-slide format) should return None, not crash or guess wrong —
    the caller (parse_pdf) is responsible for deciding the fallback.
    """
    assert extract_week_label("Just some random slide content, no dates here.") is None


def test_week_label_for_is_the_fallback_only_and_derives_from_filename():
    """
    This is ONLY used when extract_week_label() finds nothing in the PDF's
    own text. It should still produce something readable, not crash.
    """
    assert week_label_for("april_2026.pdf") == "April 2026"
    assert week_label_for("APRIL_2026.PDF") == "April 2026"


def test_clean_text_collapses_repeated_spaces():
    messy = "Camera    Calibration   is   fundamental"
    assert clean_text(messy) == "Camera Calibration is fundamental"


def test_clean_text_collapses_excess_blank_lines_but_keeps_paragraphs():
    """
    We collapse 3+ blank lines down to exactly one blank line (\\n\\n) -
    NOT down to zero. chunk.py depends on \\n\\n surviving as the
    paragraph-boundary marker, so this test protects that contract.
    """
    messy = "First paragraph.\n\n\n\n\nSecond paragraph."
    result = clean_text(messy)
    assert result == "First paragraph.\n\nSecond paragraph."


def test_clean_text_strips_leading_and_trailing_whitespace():
    assert clean_text("   hello world   \n") == "hello world"


# ---------------------------------------------------------------------------
# INTEGRATION TEST — touches real PDFs, skipped if you haven't added them yet
# ---------------------------------------------------------------------------

def _pdfs_available() -> bool:
    return PDF_DIR.exists() and any(PDF_DIR.glob("*.pdf"))


@pytest.mark.skipif(not _pdfs_available(), reason="No PDFs found in data/pdfs/ yet")
def test_parse_pdf_on_real_file_produces_sane_records():
    """
    Runs the REAL parser against whichever PDF(s) you've actually placed
    in data/pdfs/. This is your "does this actually work on my data"
    check, separate from the unit tests above that only prove the small
    pieces are correct in isolation.
    """
    pdf_path = sorted(PDF_DIR.glob("*.pdf"))[0]
    records = parse_pdf(pdf_path)

    # Basic sanity checks - not exhaustive, just "did anything go obviously wrong"
    assert len(records) > 0, "Parser returned zero slides - is the PDF empty or scanned?"

    first = records[0]
    assert first["source"] == pdf_path.name
    assert first["slide"] == 1  # first record should be slide 1 (1-indexed)
    assert len(first["text"]) > 0

    # Slide numbers should be strictly increasing and start at (or near) 1 -
    # catches accidental duplicate/out-of-order processing.
    slide_numbers = [r["slide"] for r in records]
    assert slide_numbers == sorted(slide_numbers)

    print(f"\nParsed {len(records)} slides from {pdf_path.name}")
    print(f"First slide preview: {first['text'][:120]!r}")


if __name__ == "__main__":
    # Lets you run `python tests/test_parse_pdf.py` directly too, not just pytest.
    pytest.main([__file__, "-v"])