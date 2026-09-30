"""
TESTS for eval/run_eval.py
============================
Run with:  pytest tests/test_eval.py -v
A fake retriever proves the METRIC is computed correctly. If the metric
were wrong, every recall number you report would be wrong too.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

# pyrefly: ignore [missing-import]
from run_eval import is_hit, compute_recall_at_k


def _chunk(source, label):
    return {"source": source, "slide_label": label, "text": "x"}


def _question(q="q", source="april_2026.pdf", slides=(7,)):
    return {"question": q, "expected_source": source, "expected_slides": list(slides)}


def test_is_hit_requires_matching_source_and_slide():
    q = _question()
    assert is_hit(q, _chunk("april_2026.pdf", "slide 7"))
    assert not is_hit(q, _chunk("april_2026.pdf", "slide 8"))
    assert not is_hit(q, _chunk("jan_2026.pdf", "slide 7"))  # right slide, wrong PDF


def test_is_hit_matches_when_expected_slide_is_inside_merged_range():
    assert is_hit(_question(slides=(2,)), _chunk("april_2026.pdf", "slides 1-3"))


def test_recall_counts_hits_and_misses():
    questions = [_question("found", slides=(7,)), _question("missing", slides=(50,))]
    retrieve_fn = lambda q: [_chunk("april_2026.pdf", "slide 7")]

    result = compute_recall_at_k(questions, retrieve_fn, k=5)

    assert result["hits"] == 1 and result["total"] == 2
    assert result["recall"] == 0.5
    assert result["misses"] == ["missing"]


def test_recall_only_looks_at_top_k():
    """Correct chunk at rank 4 is a miss at k=3 but a hit at k=5."""
    retrieve_fn = lambda q: [_chunk("april_2026.pdf", f"slide {n}") for n in (1, 2, 3, 7)]
    questions = [_question(slides=(7,))]

    assert compute_recall_at_k(questions, retrieve_fn, k=3)["recall"] == 0.0
    assert compute_recall_at_k(questions, retrieve_fn, k=5)["recall"] == 1.0


def test_recall_with_no_questions_is_zero_not_a_crash():
    assert compute_recall_at_k([], lambda q: [], k=5)["recall"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])