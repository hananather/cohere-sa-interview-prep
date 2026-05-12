# 0005 Session Tool Result Cache

## Decision

Use ADK before-tool and after-tool callbacks to cache successful
`search_documents` outputs in ADK session state.

## Reason

This implements cached replay, not exact invocation resume. Cached replay is the
right first step because it is small, uses ADK's callback and session systems,
and keeps Cohere's Embed, Rerank, and cited Chat path unchanged. If a later
generation step fails, retrying the same session can reuse completed retrieval
tool outputs instead of repeating the whole retrieval run.

Exact resume was considered and not chosen for the first implementation. It
would require deeper control over ADK event replay and model planning state,
which adds more demo risk than value for the interview.

## Cache Key

The cache key is a SHA-256 hash of:

- Cache schema version.
- Tool name.
- Persona id and allowed access levels.
- Normalized tool args: query, bounded `top_k`, status filter, and language.
- Index version, collection name, and corpus manifest hash.
- Cohere Embed model, embedding dimension, and Rerank model.

## Security Boundary

The cache is session scoped. It does not share results across users, personas,
or sessions. Cached results include authorized source text because the final
answer path already needs that text, but excluded-source text is stripped before
storage and audit replay.

Cross-session caching was considered and left out. It could save work across
rehearsals, but it would need a separate store, retention policy, stronger
tenant isolation, and explicit invalidation controls. That is not needed for the
current demo.

## Consequence

`answer_audit.retrieval.tool_cache` now reports cache hit and store events. A
cache hit writes the same `last_search_*` and `search_history_*` state as a live
tool call, so final grounded generation and citation audit logic do not need a
separate path.

## References

- ADK callbacks: https://adk.dev/callbacks/
- ADK state: https://adk.dev/sessions/state/
- Cohere tool use: https://docs.cohere.com/v2/docs/tool-use-overview
- Cohere RAG citations: https://docs.cohere.com/docs/rag-citations
