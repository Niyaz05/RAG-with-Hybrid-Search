"""
DAY 3 : PART B BM25 Search + vector search
Goal : Take a user's query and retrieve the most relevant chunks from your two search systems: BM25 and vector search.

                 USER QUERY
                     │
          "What is camera calibration?"
                     │
            ┌────────┴────────┐
            ↓                 ↓
        BM25 Search       Vector Search
            │                 │
            ↓                 ↓
       Top 20 chunks      Top 20 chunks
            │                 │
            └────────┬────────┘
                     ↓
                    RRF
                     ↓
                 Top 10
"""

import pickle
from pathlib import Path
from typing import Callable

import chromadb
from rank_bm25 import BM25Okapi

import sys

sys.path.insert(0, str(Path(__file__).parent.parent/"ingest"))
# pyrefly: ignore [missing-import]
from embed_index import COLLECTION_NAME, tokenize

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
BM25_PATH = Path(__file__).parent.parent / "data" / "bm25_index.pkl"

Embed_fn = Callable[[str], list[float]]

#BM25 Search
def search_bm25(query: str, bm25: BM25Okapi, chunks: list[dict], top_k: int = 20)->list[str]:
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    scored = list(zip(chunks,scores))
    scored.sort(key = lambda pair: pair[1], reverse = True)
    return [chunk["chunk_id"] for chunk, _score in scored[:top_k]]

def load_bm25_index(path: Path = BM25_PATH) -> tuple[BM25Okapi, list[dict]]:
    with open(path, "rb") as f: #rb means read binary because pickle stores python objects in binary serialization format
        data = pickle.load(f)
    return data["bm25"], data["chunks"]

#VECTOR SEARCH
def search_vector(query: str, collection, embed_fn : Embed_fn, top_k: int = 20)->list[str]:
    query_vector = embed_fn(query)
    results = collection.query(query_embeddings=[query_vector], n_results = top_k)
    return results["ids"][0]

def load_vector_index(persist_dir: Path = CHROMA_DIR):
    client = chromadb.PersistentClient(path = str(persist_dir))
    return client.get_collection(COLLECTION_NAME)
    