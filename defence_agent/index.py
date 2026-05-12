"""Local retrieval index used by the ADK search tool.

This module is intentionally thin. The actual parsing, PDF rendering,
multimodal embedding, Chroma storage, access filtering, and reranking live in
``defence_agent.retrieval``. Keeping this layer small gives the ADK agent a
clean interface without duplicating indexing logic.
"""

from __future__ import annotations

from typing import Any

from defence_agent.retrieval.chroma_index import build_index, search_index


def ensure_index_ready() -> dict[str, Any]:
    """Create the local Chroma index if it does not already exist."""

    return build_index(force=False)


def search_pages(
    *,
    query: str,
    persona_id: str,
    top_k: int,
    status_filter: str,
    language: str,
) -> dict[str, Any]:
    """Search citation-ready document pages for the current persona."""

    return search_index(
        query=query,
        persona_id=persona_id,
        top_k=top_k,
        status_filter=status_filter,
        language=language,
    )
