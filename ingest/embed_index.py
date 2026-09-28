"""
DAY 2 : EMBEDDING
Goal : BM25 + Vector(Chroma)
BM25 - traditional keyword mapping and Vector(Chroma - nomic-embed-text using ollama) - for semantic seach
"""
import json
import pickle
import re
from pathlib import Path
import chromadb
import ollama
from rank_bm25 import BM25Okapi
from typing import Callable

CHUNKS_PATH = Path(__file__).parent.parent/ "data" / "chunks.json"
CHROMA_DIR = Path(__file__).parent.parent/"data"/"chroma_db"
BM25_PATH = Path(__file__).parent.parent/"data"/"bm25_index.pkl"

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "cv_course"

EmbedFn = Callable[[str],list[float]] # this represents that it takes a string as input and outputs a list of floats

def tokenize(text: str)->list[str]:
    #turn text into lowercase words for BM25
    return re.findall(r"[a-z0-9]+",text.lower())

def embed_text(text: str)->list[float]:
    #run ollama pull nomic-embed-text in advance and ollama serve in a split terminal
    response = ollama.embeddings(model = EMBED_MODEL, prompt = text)
    return response["embedding"]
def embed_chunks(chunks: list[dict], embed_fn: EmbedFn) -> dict:
    #embed every chunk and return the four lists
    ids, embeddings, documents, metadatas = [], [], [], []

    for chunk in chunks: #each chunk is indivually processed using embed_text
        ids.append(chunk["chunk_id"])
        embeddings.append(embed_fn(chunk["text"]))
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "week": chunk["week"],
            "slide_label": chunk["slide_label"],
        }
    )
    return {"ids":ids, "embeddings":embeddings, "documents":documents, "metadatas": metadatas}

def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    #Build a BM25 index over the chunks' tokenized text
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)

def save_vector_index(chunks: list[dict], persist_dir: Path, embed_fn: EmbedFn = embed_text): #here you have given embed_text because it is a function which takes str and returns the list of floats
    client = chromadb.PersistentClient(path = str(persist_dir)) # With Persistence Chroma writes its database information to disk
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)
    data = embed_chunks(chunks, embed_fn) #each text is embedded
    collection.add(**data)
    return collection

def save_bm25_index(chunks: list[dict], path: Path):
    path.parent.mkdir(exist_ok = True)
    bm25 = build_bm25_index(chunks)
    with open (path, "wb") as f:
        pickle.dump({"bm25":bm25, "chunks": chunks}, f)
    return bm25

def main():
    chunks = json.loads(CHUNKS_PATH.read_text())
    if not chunks:
        raise SystemExit("No chunks found in chunks.json - try running chunk.py first")
    progress = {"done" : 0}
    def embed_with_progress(text: str) -> list[float]:
        progress["done"] += 1
        print(f"Embedding{progress['done']}/{len(chunks)}", end="\r")
        return embed_text(text)

    print(f"Building vector index for {len(chunks)} chunks - Ollama")
    collection = save_vector_index(chunks, CHROMA_DIR, embed_fn = embed_with_progress)
    print(f"\n Vector index = {collection.count()} chunks -> {CHROMA_DIR}")
    print(f"Building BM25 index for {len(chunks)} chunks")
    save_bm25_index(chunks, BM25_PATH)
    print(f"BM25: {len(chunks)} chunks -> {BM25_PATH}")

if __name__ == "__main__":
    main()



