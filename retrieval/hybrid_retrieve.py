"""
DAY 3 : PART D - End-to-end hybrid retrieval
                         USER QUERY
                             │
                             ▼
                 ┌─────────────────────┐
                 │ hybrid_retrieve.py  │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
        search_bm25()              search_vector()
              │                           │
              ▼                           ▼
         BM25 TOP 20                 VECTOR TOP 20
              │                           │
              └─────────────┬─────────────┘
                            ▼
                    fuse_top_k()
                            │
                            ▼
                        RRF TOP 10
                            │
                            ▼
                    build chunk objects
                            │
                            ▼
                       rerank()
                            │
                            ▼
                  CROSS-ENCODER SCORES
                       ALL 10
                            │
                            ▼
                        SORT
                            │
                            ▼
                         TOP 5
                            │
                            ▼
                    FINAL CHUNKS
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))
# pyrefly: ignore [missing-import]
from embed_index import embed_text
from fuse import fuse_top_k
from search import search_bm25, search_vector, load_bm25_index, load_vector_index
from rerank import rerank, cross_encoder_score

def build_chunk_lookup(chunks: list[dict]) -> dict[str, dict]:
    return {chunk["chunk_id"]:chunk for chunk in chunks}

def retrieve(
    query: str,
    bm25,
    bm25_chunks : list[dict],
    vector_collection,
    chunk_lookup : dict[str,dict],
    bm25_k : int = 20,
    vector_k : int = 20,
    fused_k : int = 10,
    final_k : int = 5,
) -> list[dict]:
    bm25_ids = search_bm25(query, bm25, bm25_chunks, top_k=bm25_k)
    vector_ids = search_vector(query, vector_collection, embed_fn=embed_text, top_k=vector_k)
 
    fused_ids = fuse_top_k([bm25_ids, vector_ids], top_k=fused_k)
    fused_chunks = [chunk_lookup[cid] for cid in fused_ids if cid in chunk_lookup]
    return rerank(query, fused_chunks, score_fn=cross_encoder_score, top_k=final_k)

def compare_methods(query: str, bm25, bm25_chunks, vector_collection, chunk_lookup, top_n=3):
    print(f"\n{'=' * 70}\nQUERY : {query}\n{'=' * 70}\n")
    bm25_ids = search_bm25(query, bm25, bm25_chunks, top_k = top_n)
    print(f"\n--- BM25 only (top {top_n}) ---")
    for cid in bm25_ids:
        c = chunk_lookup[cid]
        print(f" [c{c['source']}, {c['slide_label']}] {c['text'][:80]}...")

    vector_ids = search_vector(query, vector_collection, embed_fn=embed_text, top_k=top_n)
    print(f"\n--- Vector only (top {top_n}) ---")
    for cid in vector_ids:
        c = chunk_lookup[cid]
        print(f"  [{c['source']}, {c['slide_label']}] {c['text'][:80]}...")
 
    final = retrieve(query, bm25, bm25_chunks, vector_collection, chunk_lookup, final_k=top_n)
    print(f"\n--- Fused + reranked (top {top_n}) ---")
    for c in final:
        print(f"  [{c['source']}, {c['slide_label']}] {c['text'][:80]}...")


if __name__ == "__main__":
    print("Loading indexes...")
    bm25, bm25_chunks = load_bm25_index()
    vector_collection = load_vector_index()
    chunk_lookup = build_chunk_lookup(bm25_chunks)
    print(f"Loaded {len(bm25_chunks)} chunks.\n")
 
    
    # mix exact-terminology questions with conceptual/paraphrased ones.
    test_questions = [
        "What is the extrinsic matrix of a camera?",
        "Explain Adjacency, Connectivity, Region and Boundaries",
        "What does CLAHE stand for and what problem does it solve?",
        "What are Fourier Transform and Fourier Series?",
        "Explain Viola Jones Method",
    ]
 
    for q in test_questions:
        compare_methods(q, bm25, bm25_chunks, vector_collection, chunk_lookup)