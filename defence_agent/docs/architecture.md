# Architecture

## Position

Defence Agent is a production-shaped demo, not a toy chatbot.

## Runtime

- Streamlit provides the interview UI.
- FastAPI owns auth, routing, retrieval, tools, traces, evals, and metrics.
- Cohere `ClientV2` is used directly for Embed, Rerank, Chat, and Chat streaming.
- Qdrant stores vectors when available.
- SQLite stores metadata, lexical FTS, traces, evals, feedback, and sandbox runs.

## Request Path

1. FastAPI validates the demo user from `X-Demo-User`.
2. Input safety checks inspect length and prompt-injection patterns.
3. Router selects the workflow using deterministic rules first.
4. Tools run with Pydantic schemas and policy checks.
5. Retrieval applies ACL filters before lexical or vector search.
6. Cohere Rerank sorts candidate chunks.
7. The workflow generates a cited answer or degrades safely.
8. Trace spans and metrics are recorded.

## Why Direct Cohere SDK

- The interview is for Cohere, so Cohere should be visible in the implementation.
- Direct `ClientV2` calls keep the architecture simpler and easier to defend.
- ADK-style concepts are still present: router, workflows, tools, traces, and evals.
