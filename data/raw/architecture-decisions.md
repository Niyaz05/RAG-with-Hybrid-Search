# Architecture Decision Records

## ADR-014: Hybrid Retrieval Instead of Dense-Only

We evaluated dense-only retrieval against a hybrid dense + BM25 approach
during the Q2 platform redesign. Dense-only retrieval underperformed on
queries containing exact identifiers — API error codes, config keys, and
customer account IDs — because embedding models compress these tokens into
semantic neighborhoods that don't preserve exact-match precision.

We adopted Reciprocal Rank Fusion to combine dense and BM25 result lists,
weighted 0.7 dense / 0.3 sparse by default. This weighting is configurable
per customer workspace, since some customers' corpora (legal contracts) are
far more identifier-heavy than others (marketing copy).

## ADR-019: Reranker Placement After Fusion

The cross-encoder reranker runs after RRF fusion, on the top 20 fused
candidates, rather than reranking the dense and sparse lists separately
before fusion. Reranking after fusion reduced end-to-end query latency by
roughly 40ms compared to reranking both lists independently, since the
reranker only scores 20 candidates instead of 40.

## ADR-022: Compute Pool Sharing Between Ingestion and Query

Bulk re-indexing jobs and query-time reranking share the same GPU compute
pool. This was a deliberate cost tradeoff rather than an oversight: running
separate dedicated pools for ingestion and query would have roughly doubled
GPU spend for workloads that are rarely both under heavy load
simultaneously. The known tradeoff is that a large bulk re-index can degrade
query latency during the re-index window, which the on-call team should
expect to see reflected in the query latency dashboards.
