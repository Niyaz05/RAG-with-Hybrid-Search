
"""
DAY 3 :PART C - Once you have RRF top 10 chunks, Cross-encoder gives a relevance score for each chunk and take the top 5
                 RRF TOP 10 CHUNKS
                        │
                        │
                        ▼
              ┌───────────────────┐
              │   RERANK FUNCTION │
              │                   │
              │ query + 10 chunks │
              └─────────┬─────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   Cross-Encoder     │
             │                     │
             │ Query + Chunk Text  │
             │       together      │
             └──────────┬──────────┘
                        │
                        ▼
                Calculate a
              relevance score
                for EACH chunk
                        │
             ┌──────────┼──────────┐
             │          │          │
             ▼          ▼          ▼
          Chunk 1    Chunk 2    Chunk 3   ... Chunk 10
            8.7        5.2        9.1          3.4
             │          │          │             │
             └──────────┴──────────┴─────────────┘
                        │
                        ▼
                SORT BY SCORE
                 highest → lowest
                        │
                        ▼
               ┌────────────────┐
               │     TOP 5      │
               │                │
               │  1. chunk_3    │
               │  2. chunk_1    │
               │  3. ...        │
               │  4. ...        │
               │  5. ...        │
               └───────┬────────┘
                       │
                       ▼
                FINAL RELEVANT
                     CHUNKS
                       │
                       ▼
                  RAG / LLM
                  ANSWERING
"""
from typing import Callable
Scorefn = Callable[[str,str], float]

def rerank(query: str, candidates : list[dict], score_fn: Scorefn, top_k : int = 5) -> list[dict]:
    scored = [(chunk, score_fn(query, chunk["text"])) for chunk in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _score in scored[:top_k]]

_model_cache = {}
from sentence_transformers import CrossEncoder
def cross_encoder_score(query: str, document_text: str) -> float:
    if "model" not in _model_cache:
        _model_cache["model"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
 
    model = _model_cache["model"]
    # CrossEncoder.predict expects a list of (query, doc) pairs and
    # returns a list of scores - we're only scoring one pair here.
    score = model.predict([(query, document_text)])[0]
    return float(score)

