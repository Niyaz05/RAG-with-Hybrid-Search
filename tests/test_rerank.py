"""
TESTS for retrieval/rerank.py
================================
Run with:  pytest tests/test_rerank.py -v

Same three-tier pattern as Day 2's embed tests:
  1. Logic tests with a FAKE score_fn - instant, no model needed.
  2. One real integration test using the actual cross-encoder model -
     skipped automatically if it can't be loaded (no internet access to
     Hugging Face, or model not cached yet).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))

from retrieval.rerank import rerank, cross_encoder_score


# ---------------------------------------------------------------------------
# Logic tests — fake score_fn, no model
# ---------------------------------------------------------------------------

def _fake_score_fn(query: str, document_text: str) -> float:
    """
    A deterministic fake scorer: score = how many query words appear in
    the document. Good enough to prove rerank()'s SORTING/TRUNCATING
    logic works, without needing real semantic understanding.
    """
    query_words = set(query.lower().split())
    doc_words = set(document_text.lower().split())
    return float(len(query_words & doc_words))


def test_rerank_sorts_by_score_descending():
    candidates = [
        {"chunk_id": "c1", "text": "optical flow measures pixel motion"},
        {"chunk_id": "c2", "text": "camera calibration estimates intrinsic parameters"},
        {"chunk_id": "c3", "text": "video compression reduces file size"},
    ]
    query = "camera calibration intrinsic"

    result = rerank(query, candidates, score_fn=_fake_score_fn, top_k=3)

    # c2 shares the most words with the query -> should come first
    assert result[0]["chunk_id"] == "c2"


def test_rerank_truncates_to_top_k():
    candidates = [
        {"chunk_id": f"c{i}", "text": f"chunk number {i} about camera calibration"}
        for i in range(10)
    ]
    result = rerank("camera calibration", candidates, score_fn=_fake_score_fn, top_k=3)

    assert len(result) == 3


def test_rerank_preserves_full_chunk_dicts_not_just_ids():
    """
    rerank() should return the ORIGINAL chunk dicts (with all their
    metadata intact), just reordered - not strip them down to bare ids.
    Downstream generation needs source/slide_label for citations.
    """
    candidates = [
        {"chunk_id": "c1", "text": "camera calibration", "source": "jan_2026.pdf", "slide_label": "slide 3"},
    ]
    result = rerank("camera", candidates, score_fn=_fake_score_fn, top_k=1)

    assert result[0]["source"] == "jan_2026.pdf"
    assert result[0]["slide_label"] == "slide 3"


def test_rerank_handles_empty_candidates():
    assert rerank("anything", [], score_fn=_fake_score_fn, top_k=5) == []


# ---------------------------------------------------------------------------
# Real integration test — needs the actual cross-encoder model
# ---------------------------------------------------------------------------

def _cross_encoder_available() -> bool:
    try:
        cross_encoder_score("test query", "test document")
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _cross_encoder_available(),
    reason="Cross-encoder model not available (needs Hugging Face download on first use)",
)
def test_real_cross_encoder_prefers_relevant_document():
    """
    Confirms the REAL model scores an on-topic document higher than an
    unrelated one for a specific query - a basic sanity check, not a
    thorough evaluation (that's what Day 4's eval harness is for).
    """
    query = "What is the extrinsic matrix of a camera?"
    relevant_doc = "The extrinsic matrix describes the camera's position and orientation in the world coordinate frame."
    irrelevant_doc = "Optical character recognition converts scanned documents into searchable text."

    relevant_score = cross_encoder_score(query, relevant_doc)
    irrelevant_score = cross_encoder_score(query, irrelevant_doc)

    assert relevant_score > irrelevant_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])