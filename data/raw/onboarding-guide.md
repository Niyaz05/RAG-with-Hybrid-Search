# New Engineer Onboarding Guide

## First Week

New engineers on the platform team should complete local environment setup
using the `setup.sh` script in the `infra` repository before their first
day, since it takes roughly 45 minutes to provision local Kubernetes and
seed sample data.

Access to the production dashboard requires a manager-approved request
through the internal access portal. This typically takes 1-2 business days
to be granted, so submit the request in your first week rather than waiting
until you need production access.

## Codebase Orientation

The ingestion pipeline lives in `services/ingestion`, the retrieval and
reranking service lives in `services/retrieval`, and the generation and
citation-verification service lives in `services/generation`. Each service
has its own on-call rotation; new engineers join a rotation only after
completing their first 30 days and shadowing at least one incident.

## Who to Ask

For questions about chunking strategy or embedding model choice, the
retrieval team owns that decision. For questions about rate limiting or
API authentication, the platform team owns that. For incident response
process questions, refer to the incident runbook first before asking in
the on-call channel.
