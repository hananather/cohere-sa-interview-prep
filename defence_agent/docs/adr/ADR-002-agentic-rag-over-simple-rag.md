# ADR-002: Use Agentic RAG Over Simple RAG

## Decision

Use route-specific workflows instead of one generic RAG chain.

## Reason

The demo needs version comparison, table analysis, restricted access handling, prompt-injection defense, and human review. Those are workflow problems, not just retrieval problems.

## Consequence

The system has more moving parts, but the trace makes those parts explainable.
