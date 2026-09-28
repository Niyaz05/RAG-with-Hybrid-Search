"""
DAY 3 : PART A Fuse
Goal : combine BM25 and vector-search ranking of same chunk_ids into one ranking

                    BM25 SEARCH
                         │
                         │ top 20 chunk IDs
                         ▼
              ["c42", "c17", "c81", ...]
                         │
                         │
                         │
                    VECTOR SEARCH
                         │
                         │ top 20 chunk IDs
                         ▼
              ["c17", "c93", "c42", ...]
                         │
                         │
                         └──────────────┐
                                        │
                                        ▼
                              reciprocal_rank_fusion()
                                        │
                                        ▼
                         Look at RANK, not scores
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                    BM25 ranking                Vector ranking
                         │                             │
                  c42 → rank 1                  c17 → rank 1
                  c17 → rank 2                  c42 → rank 2
                  c81 → rank 3                  c93 → rank 3
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                              Calculate RRF score
                                        │
                         score = 1 / (k + rank)
                                        │
                              k = 60 by default
                                        │
                                        ▼
                         Add scores for same chunks
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                    c42 appears twice           c81 appears once
                         │                             │
                    BM25: 1/61                 BM25: 1/63
                    Vector: 1/62               Vector: 0
                         │                             │
                         ▼                             ▼
                    total score                 total score
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                              SORT BY RRF SCORE
                              highest → lowest
                                        │
                                        ▼
                         [
                           ("c42", score),
                           ("c17", score),
                           ("c93", score),
                           ...
                         ]
                                        │
                                        ▼
                              fuse_top_k()
                                        │
                                        ▼
                               Take first 10
                                        │
                                        ▼
                         Remove the RRF scores
                                        │
                                        ▼
                    ["c42", "c17", "c93", ...]
                                        │
                                        ▼
                                  RRF TOP 10
                                        │
                                        ▼
                              CROSS-ENCODER

                              
THE CORE PROBLEM RRF SOLVES:
BM25 scores and cosine-similarity scores live on completely different,
incompatible scales. BM25 scores are unbounded and depend on your corpus
size/vocabulary (a score of "8.3" means nothing without context). Cosine
similarity is bounded between -1 and 1. You CANNOT just average them —
whichever one happens to have bigger numbers would dominate for no
principled reason.
 
RRF'S TRICK: ignore the scores entirely. Only look at each doc's RANK
POSITION (1st, 2nd, 3rd...) within each method's results, and combine
THOSE with a simple formula:
 
    score(doc) = sum over each ranking method of:  1 / (k + rank(doc))
 
  - A doc ranked #1 by a method contributes a LARGE amount (1/(k+1)).
  - A doc ranked #50 by a method contributes very little (1/(k+50)).
  - A doc that doesn't appear in a method's results at all contributes 0
    from that method (but can still win overall if the OTHER method
    ranked it highly).
  - `k` (commonly 60) is a damping constant — it softens how much rank #1
    dominates over rank #2 vs rank #2 over rank #3. Larger k = flatter,
    more egalitarian scoring; smaller k = rank #1 dominates harder.
"""

K_DEFAULT = 60

def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = K_DEFAULT) -> list[tuple[str,float]]:
    scores: dict[str,float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start = 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k+rank) #scores.get checks if a score is already given by a method, if it does it add the new score and if not it starts with 0.0
    
    #sort by fuse score in descending order
    fused = sorted(scores.items(), key = lambda pair : pair[1], reverse = True)#pairs are in ("chunk_id", score) so by doing pair[1] we are fetching the score for sorting
    return fused

def fuse_top_k(ranked_lists: list[list[str]], top_k : int = 10, k: int = K_DEFAULT) ->list[str]:
    fused = reciprocal_rank_fusion(ranked_lists, k=k)
    return [chunk_id for chunk_id, _score in fused[:top_k]] #_score '_' is a python naming convention for variables that are not used
    
