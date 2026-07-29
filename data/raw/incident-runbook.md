# Incident Response Runbook

## Severity Levels

Aperture Cloud classifies incidents into three severity levels.

Sev1 incidents involve full platform outage or data loss and require paging
the on-call engineer immediately, with a status page update within 15
minutes. Sev2 incidents involve partial degradation, such as elevated error
rates on the `/v1/query` endpoint, and require a status page update within
1 hour. Sev3 incidents are minor and do not require a public status update.

## Ingestion Pipeline Failures

If document ingestion jobs are failing in bulk, first check the indexing
queue depth in the internal dashboard. A queue depth above 5000 pending jobs
typically indicates the embedding worker pool has crashed rather than a
downstream API issue.

Restart the embedding worker pool with:

    kubectl rollout restart deployment/embedding-worker -n ingestion

If failures persist after a restart, check whether the upstream embeddings
provider is reporting elevated error rates on their status page. This has
historically been the root cause in 3 of the last 5 ingestion incidents.

## Query Latency Spikes

Elevated p99 latency on `/v1/query` is most often caused by the reranker
step, not the initial retrieval. Check the reranker service's request queue
before assuming the vector database is the bottleneck — this is a common
misdiagnosis during on-call incidents. The vector database has historically
only been the root cause in query latency incidents when the queue depth
dashboard also shows an ingestion backlog at the same time, since bulk
re-indexing competes for the same compute pool as query-time reranking.

## Rollback Procedure

Any deploy causing a Sev1 or Sev2 incident should be rolled back
immediately rather than patched forward. Use the deployment dashboard's
"Rollback to previous" button, which reverts to the last known-good image
within roughly 2 minutes.
