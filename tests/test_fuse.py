"""
TESTS for retrieval/fuse.py
=============================
Run with:  pytest tests/test_fuse.py -v

These are all pure logic tests using tiny made-up chunk_id lists like
["a", "b", "c"] - no PDFs, no embeddings, no BM25, no model downloads.
That's the payoff of keeping reciprocal_rank_fusion() as a pure function
(strings and floats in, strings and floats out).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))

from retrieval.fuse import reciprocal_rank_fusion, fuse_top_k


def test_doc_ranked_first_by_both_methods_scores_highest():
    bm25_ranked = ["a", "b", "c"]
    vector_ranked = ["a", "c", "b"]

    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked])
    top_id, top_score = fused[0]

    assert top_id == "a"  # ranked #1 by both methods


def test_doc_present_in_only_one_list_still_gets_a_score():
    """
    A doc that ONLY BM25 found (e.g. an exact-terminology match vector
    search missed) should still make it into the fused ranking with a
    non-zero score - it just won't score as high as a doc both methods agree on.
    """
    bm25_ranked = ["a", "b"]
    vector_ranked = ["c", "d"]  # "b" never appears here

    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked])
    fused_ids = [chunk_id for chunk_id, _score in fused]

    assert "b" in fused_ids
    scores = dict(fused)
    assert scores["b"] > 0.0


def test_doc_found_by_both_methods_outscores_doc_found_by_only_one():
    """
    Core hybrid-search promise: a chunk that both BM25 AND vector search
    agree on (even at moderate ranks) should beat a chunk only one method
    found, all else being roughly equal.
    """
    bm25_ranked = ["shared", "bm25_only"]
    vector_ranked = ["shared", "vector_only"]

    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked])
    scores = dict(fused)

    assert scores["shared"] > scores["bm25_only"]
    assert scores["shared"] > scores["vector_only"]


def test_larger_k_flattens_the_score_distribution():
    """
    LEARNING NOTE: as k grows, 1/(k+rank) values for different ranks get
    closer together (the formula becomes less sensitive to WHICH rank a
    doc is at). This test just confirms that relationship holds, as a
    sanity check on the formula itself.
    """
    ranked = ["a", "b", "c"]

    fused_small_k = dict(reciprocal_rank_fusion([ranked], k=1))
    fused_large_k = dict(reciprocal_rank_fusion([ranked], k=1000))

    gap_small_k = fused_small_k["a"] - fused_small_k["c"]
    gap_large_k = fused_large_k["a"] - fused_large_k["c"]

    assert gap_small_k > gap_large_k


def test_empty_ranked_lists_produce_no_results():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_fuse_top_k_truncates_correctly():
    bm25_ranked = ["a", "b", "c", "d", "e"]
    vector_ranked = ["a", "b", "c", "d", "e"]

    top_2 = fuse_top_k([bm25_ranked, vector_ranked], top_k=2)

    assert len(top_2) == 2
    assert top_2 == ["a", "b"]  # both methods agree on the same order here


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])