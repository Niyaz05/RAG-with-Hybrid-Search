"""
TESTS for retrieval/search.py
================================
Run with:  pytest tests/test_search.py -v

BM25 tests: build an in-memory BM25 index directly (same as Day 2's
tests), no pickle file needed.
Vector tests: use chromadb.EphemeralClient() (in-memory, no disk writes)
with a FAKE embed_fn - same dependency-injection trick as Day 2, so no
Ollama call happens here.
"""

import sys
from pathlib import Path

import chromadb
import pytest
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from retrieval.search import search_bm25, search_vector
from ingest.embed_index import tokenize


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

def _sample_chunks():
    return [
        {"chunk_id": "c0001", "text": "Camera calibration estimates intrinsic parameters."},
        {"chunk_id": "c0002", "text": "Optical flow measures pixel motion between frames."},
        {"chunk_id": "c0003", "text": "OCR converts scanned documents into searchable text."},
        {"chunk_id": "c0004", "text": "Video compression reduces redundancy between frames."},
    ]


def test_search_bm25_returns_chunk_ids_not_text():
    chunks = _sample_chunks()
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    results = search_bm25("camera intrinsic parameters", bm25, chunks, top_k=2)

    assert results == ["c0001", "c0004"] or results[0] == "c0001"
    assert all(isinstance(r, str) for r in results)


def test_search_bm25_respects_top_k():
    chunks = _sample_chunks()
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    results = search_bm25("frames", bm25, chunks, top_k=2)

    assert len(results) == 2


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def _fake_embed_fn(text: str) -> list[float]:
    """Same deterministic fake used in Day 2's embed tests."""
    words = text.lower().split()
    return [
        float(sum(1 for w in words if "camera" in w or "calibrat" in w)),
        float(sum(1 for w in words if "optical" in w or "flow" in w)),
        float(len(text)),
    ]


def test_search_vector_returns_top_k_ids():
    client = chromadb.EphemeralClient()  # in-memory only, nothing touches disk
    collection = client.create_collection("test_collection")

    chunks = _sample_chunks()
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[_fake_embed_fn(c["text"]) for c in chunks],
        documents=[c["text"] for c in chunks],
    )

    results = search_vector("camera calibration", collection, embed_fn=_fake_embed_fn, top_k=2)

    assert len(results) == 2
    assert results[0] == "c0001"  # our fake embedding makes this the closest match


if __name__ == "__main__":
    pytest.main([__file__, "-v"])