"""
TESTS for ingest/embed_index.py
=================================
Run with:  pytest tests/test_embed_index.py -v

LEARNING NOTE — three tiers of test here, cheapest/most-reliable first:
1. tokenize() - pure function, zero dependencies.
2. embed_chunks() / build_bm25_index() - use a FAKE embed_fn instead of
   real Ollama, so they run instantly and don't require Ollama to be
   running. This is the dependency-injection trick from embed_index.py.
3. ONE real integration test that actually calls Ollama - skipped
   automatically if Ollama isn't reachable, same pattern as Day 1's
   PDF-availability skip.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))


# pyrefly: ignore [missing-import]
from embed_index import tokenize, embed_chunks, build_bm25_index, embed_text


# ---------------------------------------------------------------------------
# tokenize() — pure function
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_splits_on_words():
    assert tokenize("Camera Calibration") == ["camera", "calibration"]


def test_tokenize_handles_punctuation_and_numbers():
    # Punctuation is dropped; numbers are kept as their own tokens.
    result = tokenize("RRF uses k=60 (a constant).")
    assert result == ["rrf", "uses", "k", "60", "a", "constant"]


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("") == []


# ---------------------------------------------------------------------------
# embed_chunks() — uses a FAKE embed_fn, no Ollama required
# ---------------------------------------------------------------------------

def _fake_embed_fn(text: str) -> list[float]:
    """
    A deterministic fake embedding: just returns [length of text, 0.0, 0.0].
    We don't care that it's not a REAL semantic embedding - this test is
    only checking that embed_chunks() wires ids/embeddings/documents/
    metadatas together correctly, not that the embeddings are good.
    """
    return [float(len(text)), 0.0, 0.0]


def _sample_chunks():
    return [
        {"chunk_id": "c0001", "source": "jan_2026.pdf", "week": "5-9 Jan 2026",
         "slide_label": "slide 1", "text": "Camera calibration basics."},
        {"chunk_id": "c0002", "source": "jan_2026.pdf", "week": "5-9 Jan 2026",
         "slide_label": "slide 2", "text": "Intrinsic and extrinsic parameters."},
    ]


def test_embed_chunks_returns_aligned_parallel_lists():
    chunks = _sample_chunks()
    result = embed_chunks(chunks, embed_fn=_fake_embed_fn)

    assert result["ids"] == ["c0001", "c0002"]
    assert result["documents"] == [c["text"] for c in chunks]
    assert len(result["embeddings"]) == 2
    assert len(result["metadatas"]) == 2


def test_embed_chunks_preserves_metadata_fields():
    chunks = _sample_chunks()
    result = embed_chunks(chunks, embed_fn=_fake_embed_fn)

    assert result["metadatas"][0] == {
        "source": "jan_2026.pdf",
        "week": "5-9 Jan 2026",
        "slide_label": "slide 1",
    }


def test_embed_chunks_uses_the_injected_fn_not_a_hardcoded_one():
    """
    This is the test that PROVES dependency injection is actually working:
    if embed_chunks() secretly called the real embed_text() instead of the
    embed_fn parameter, this test would try to hit Ollama and likely fail
    or hang in a CI/sandbox environment with no Ollama running.
    """
    chunks = [_sample_chunks()[0]]
    result = embed_chunks(chunks, embed_fn=_fake_embed_fn)

    # Our fake embed_fn returns [len(text), 0.0, 0.0] - verify that's what came back.
    expected_len = float(len(chunks[0]["text"]))
    assert result["embeddings"][0] == [expected_len, 0.0, 0.0]


# ---------------------------------------------------------------------------
# build_bm25_index() — no Ollama needed at all, pure CPU
# ---------------------------------------------------------------------------

def test_bm25_index_ranks_relevant_chunk_higher():
    """
    LEARNING NOTE — why 4 chunks, not 2: BM25's IDF formula is
    log((N - n + 0.5) / (n + 0.5)), where N = total docs, n = docs
    containing the term. With only 2 total docs and a term in exactly 1
    of them, that formula evaluates to log(1.0) = 0 for every term -
    correct behavior (a term in half a 2-doc corpus isn't discriminative
    at all), but useless for demonstrating ranking. A slightly bigger,
    more realistic corpus avoids that degenerate case.
    """
    chunks = [
        {"chunk_id": "c0001", "text": "Camera calibration estimates intrinsic parameters."},
        {"chunk_id": "c0002", "text": "Optical flow measures pixel motion between frames."},
        {"chunk_id": "c0003", "text": "OCR converts scanned documents into searchable text."},
        {"chunk_id": "c0004", "text": "Video compression reduces redundancy between frames."},
    ]
    bm25 = build_bm25_index(chunks)

    query_tokens = tokenize("intrinsic parameters calibration")
    scores = bm25.get_scores(query_tokens)

    # chunk 0 (calibration) should clearly outscore the other three, which
    # share none of the query's distinctive terms.
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[0] > scores[3]


# ---------------------------------------------------------------------------
# INTEGRATION TEST — actually calls Ollama, skipped if it's not reachable
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """
    Try a trivial real embedding call; if it raises (Ollama not running,
    model not pulled, etc.), we treat Ollama as unavailable and skip
    rather than fail the whole suite.
    """
    try:
        embed_text("test")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not reachable / model not pulled")
def test_real_embedding_has_expected_shape():
    """
    Confirms the REAL nomic-embed-text call returns a non-trivial vector.
    We don't assert exact values (embeddings aren't meant to be checked
    that way) - just that we got a real, reasonably-sized float vector back.
    """
    vector = embed_text("Camera calibration bridges 2D images and 3D space.")
    assert isinstance(vector, list)
    assert len(vector) > 100  # nomic-embed-text produces 768-dim vectors
    assert all(isinstance(x, float) for x in vector[:5])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])