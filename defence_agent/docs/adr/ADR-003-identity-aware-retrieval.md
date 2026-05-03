# ADR-003: Use Identity-Aware Retrieval

## Decision

Apply ACL filters before lexical search, vector search, rerank, and generation.

## Reason

The model must never see unauthorized chunks. Post-generation filtering is too late.

## Consequence

The same restricted query returns different results for `planning_analyst` and `planning_lead`.
