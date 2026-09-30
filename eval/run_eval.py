"""
DAY 4 — PART B: Retrieval evaluation (recall@k)
=================================================
For each question in eval_questions.json we know WHICH slide(s) contain the
answer. Recall@k = the fraction of questions where at least one retrieved
chunk in the top k comes from an expected slide.

This measures RETRIEVAL only, separate from whether the LLM's answer is
good. Most RAG failures are retrieval failures, so it's the first thing to
measure. Run it before/after every pipeline change (chunk size, RRF k,
rerank cutoff) to catch regressions.

retrieve_fn is injected, so the metric logic is testable with a fake retriever.
"""

import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "generation"))  # for slide_numbers
# pyrefly: ignore [missing-import]
from prompt import slide_numbers

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"

RetrieveFn = Callable[[str], list[dict]]


def is_hit(question: dict, chunk: dict) -> bool:
    """A chunk is a hit if it's from the expected PDF and overlaps an expected slide."""
    if chunk["source"].lower() != question["expected_source"].lower():
        return False
    return bool(set(question["expected_slides"]) & slide_numbers(chunk["slide_label"]))


def compute_recall_at_k(questions: list[dict], retrieve_fn: RetrieveFn, k: int = 5) -> dict:
    """Returns {"recall": float, "hits": int, "total": int, "misses": [question texts]}."""
    hits, misses = 0, []
    for q in questions:
        top_k = retrieve_fn(q["question"])[:k]
        if any(is_hit(q, chunk) for chunk in top_k):
            hits += 1
        else:
            misses.append(q["question"])
    total = len(questions)
    return {"recall": hits / total if total else 0.0, "hits": hits, "total": total, "misses": misses}


def main():
    # Real-pipeline imports live here so the pure logic above imports cleanly in tests.
    sys.path.insert(0, str(ROOT / "retrieval"))
    sys.path.insert(0, str(ROOT / "ingest"))
    # pyrefly: ignore [missing-import]
    from embed_index import embed_text
    # pyrefly: ignore [missing-import]
    from search import search_bm25, search_vector, load_bm25_index, load_vector_index
    # pyrefly: ignore [missing-import]
    from hybrid_retrieve import retrieve, build_chunk_lookup

    questions = json.loads(QUESTIONS_PATH.read_text())
    bm25, chunks = load_bm25_index()
    collection = load_vector_index()
    lookup = build_chunk_lookup(chunks)

    def bm25_only(q):
        return [lookup[i] for i in search_bm25(q, bm25, chunks, top_k=5)]

    def vector_only(q):
        return [lookup[i] for i in search_vector(q, collection, embed_text, top_k=5)]

    def hybrid(q):
        return retrieve(q, bm25, chunks, collection, lookup, final_k=5)

    print(f"Evaluating {len(questions)} questions at recall@5\n")
    for name, fn in [("BM25 only", bm25_only), ("Vector only", vector_only), ("Hybrid + rerank", hybrid)]:
        r = compute_recall_at_k(questions, fn, k=5)
        print(f"{name:<16} recall@5 = {r['recall']:.0%}  ({r['hits']}/{r['total']})")
        for m in r["misses"]:
            print(f"    missed: {m}")


if __name__ == "__main__":
    main()