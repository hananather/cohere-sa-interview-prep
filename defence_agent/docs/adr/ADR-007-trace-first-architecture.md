# ADR-007: Use Trace-First Architecture

## Decision

Every major step writes a span.

## Reason

Agent quality depends on the trajectory, not only the final answer. Traces make failures debuggable and auditable.

## Consequence

The Streamlit Trace tab can explain routing, retrieval, reranking, tools, citations, safety, and latency.
