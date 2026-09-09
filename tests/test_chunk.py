"""
TESTS for ingest/chunk.py
===========================
Run with:  pytest tests/test_chunk.py -v

LEARNING NOTE — why synthetic/fake data instead of your real chunks.json?
Real data is messy and unpredictable - great for catching SURPRISES, bad
for proving a SPECIFIC rule works. Here we hand-craft tiny fake page lists
where we know exactly what SHOULD happen (this page is thin, it MUST
merge; this page is huge, it MUST split) - then assert that it does. This
is the standard way to test data-transformation logic.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from chunk import build_chunks, split_dense_page, MIN_CHARS, MAX_CHARS


# ---------------------------------------------------------------------------
# split_dense_page() — pure function, easiest to test directly
# ---------------------------------------------------------------------------

def test_split_dense_page_keeps_short_text_as_single_piece():
    text = "This is a short slide.\n\nIt has two paragraphs."
    pieces = split_dense_page(text, max_chars=1400)
    assert len(pieces) == 1
    assert pieces[0] == text


def test_split_dense_page_splits_on_paragraph_boundaries():
    # Build 3 paragraphs that individually fit but together exceed max_chars.
    para = "Camera calibration bridges 2D images and 3D space. " * 5  # ~265 chars
    text = "\n\n".join([para, para, para])  # ~800 chars total
    pieces = split_dense_page(text, max_chars=400)

    assert len(pieces) > 1, "Expected the long text to be split into multiple pieces"
    for piece in pieces:
        # Each piece should respect the max_chars budget (with small slack
        # since we don't cut mid-paragraph).
        assert len(piece) <= 400 + 5


def test_split_dense_page_hard_splits_a_single_oversized_paragraph():
    """
    Edge case: ONE paragraph with no internal \\n\\n breaks that is itself
    longer than max_chars. There's no paragraph boundary to exploit, so
    the function must fall back to a character-count split rather than
    returning one giant unsplit chunk.
    """
    huge_single_paragraph = "word " * 500  # one paragraph, no \n\n at all
    pieces = split_dense_page(huge_single_paragraph, max_chars=300)

    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)


# ---------------------------------------------------------------------------
# build_chunks() — the merge + split orchestration logic
# ---------------------------------------------------------------------------

def _page(source, week, slide, text):
    """Small helper to build a fake parsed-page dict without repeating keys."""
    return {"source": source, "week": week, "slide": slide, "text": text}


def test_thin_title_slide_merges_into_next_slide():
    """
    A near-empty title slide (like your real cover pages: 'CSET340 /
    Advanced Computer Vision / 6th-10th April 2026') should NOT become
    its own chunk - it should merge forward into the next real slide.
    """
    thin_title = _page("april_2026.pdf", "Week 4", 1, "CSET340\nAdvanced CV\n6-10 April")
    real_content = _page(
        "april_2026.pdf", "Week 4", 2,
        "Camera Calibration\n\n" + ("Camera calibration bridges 2D and 3D. " * 10),
    )

    chunks = build_chunks([thin_title, real_content])

    # Expect exactly ONE chunk covering both slides, not two separate chunks.
    assert len(chunks) == 1
    assert chunks[0]["slide_label"] == "slides 1-2"
    assert "CSET340" in chunks[0]["text"]          # thin slide's text preserved
    assert "Camera calibration" in chunks[0]["text"]  # merged with real content


def test_thin_slide_does_not_merge_across_different_pdfs():
    """
    If a thin slide is the LAST page of one PDF and the next page in our
    list belongs to a DIFFERENT PDF, they must NOT merge - that would mix
    content from two different weeks into one chunk with a nonsense
    citation.
    """
    thin_last_page_of_march = _page("march_2026.pdf", "Week 3", 20, "End of week 3.")
    first_page_of_april = _page(
        "april_2026.pdf", "Week 4", 1,
        "CSET340\n\n" + ("Advanced Computer Vision course intro. " * 10),
    )

    chunks = build_chunks([thin_last_page_of_march, first_page_of_april])

    sources = {c["source"] for c in chunks}
    assert sources == {"march_2026.pdf", "april_2026.pdf"}, (
        "Expected two separate chunks with distinct sources, not one merged chunk"
    )


def test_dense_slide_gets_split_into_multiple_chunks():
    dense_text = ("Extracting intrinsic and extrinsic parameters. " * 40 + "\n\n"
                  + "We know the projection matrix P. " * 30)
    dense_page = _page("april_2026.pdf", "Week 4", 20, dense_text)

    chunks = build_chunks([dense_page])

    assert len(chunks) > 1, "Expected the dense slide to split into multiple chunks"
    for c in chunks:
        assert c["slide_label"] == "slide 20"  # all pieces still cite the same slide


def test_chunk_ids_are_unique_and_sequential():
    pages = [
        _page("jan_2026.pdf", "Week 1", 1, "A" * 200),
        _page("jan_2026.pdf", "Week 1", 2, "B" * 200),
        _page("jan_2026.pdf", "Week 1", 3, "C" * 200),
    ]
    chunks = build_chunks(pages)
    ids = [c["chunk_id"] for c in chunks]

    assert len(ids) == len(set(ids)), "chunk_ids must be unique"
    assert ids == sorted(ids), "chunk_ids should be assigned in order"


def test_empty_input_produces_no_chunks():
    """Edge case: an empty page list shouldn't crash - just return nothing."""
    assert build_chunks([]) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])