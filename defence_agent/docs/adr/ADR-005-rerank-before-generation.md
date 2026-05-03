# ADR-005: Use Hybrid Retrieval And Rerank

## Decision

Merge lexical and vector candidates, then rerank before generation.

## Reason

Lexical search catches exact terms, vector search catches semantic matches, and Cohere Rerank improves final context ordering.

## Consequence

The trace records lexical, vector, hybrid, and rerank scores for audit.
