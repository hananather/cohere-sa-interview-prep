"""Tools exposed to the ADK agent.

The canonical agent intentionally has one tool. The model can call this tool
multiple times for agentic RAG instead of relying on a large tool portfolio.
"""

from __future__ import annotations

import os
from typing import Any

from defence_agent.auth.context import DEFAULT_PERSONA_ID, DEMO_USERS
from defence_agent.auth.policy import policy_engine
from defence_agent.tool_state import record_search_documents_state

from .index import search_pages

MAX_TOOL_TOP_K = 24
MIN_TOOL_TOP_K = 8


def _persona_id(tool_context: Any | None = None) -> str:
    """Resolve the active persona from ADK session state.

    ADK injects ``tool_context`` when the model calls this tool. The persona is
    session state, not a model-controlled argument, so the model cannot ask the
    tool to run as a different user. The environment fallback keeps direct CLI
    and unit-test usage simple.
    """

    state = getattr(tool_context, "state", {}) if tool_context is not None else {}
    requested = state.get("persona_id") or os.getenv("DEFTECH_ADK_PERSONA", DEFAULT_PERSONA_ID)
    return requested if requested in DEMO_USERS else DEFAULT_PERSONA_ID


def _bounded_top_k(top_k: int) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        value = 8
    return min(max(value, MIN_TOOL_TOP_K), MAX_TOOL_TOP_K)


def search_documents(
    query: str,
    top_k: int = 8,
    status_filter: str = "approved",
    language: str = "any",
    tool_context: Any | None = None,
) -> dict[str, Any]:
    """Search authorized DefTech doctrine pages in local Chroma.

    Use this for questions about NATO and Canadian defence PDFs, AI strategy,
    strategic concepts, defence policy, sensitive synthetic annexes, current
    guidance, and bilingual English/French sources.

    The active persona is read from ADK session state, not from model-controlled
    parameters. Excluded source text is never returned.

    Args:
        query: The search phrase for the doctrine corpus.
        top_k: Number of final reranked source pages to return. Clamped to
            the inclusive range 8..24.
        status_filter: One of "approved", "draft", "superseded", or "any".
            Use "approved" for current guidance. Use "any" for comparison or
            currentness questions.
        language: One of "any", "en", or "fr". Use "any" unless the user asks
            for a specific source language.

    Returns:
        Authorized source pages with citation IDs, source metadata, rerank
        scores, filters applied, and excluded-source metadata. Excluded source
        text is never returned.
    """

    persona_id = _persona_id(tool_context)
    auth = DEMO_USERS[persona_id]
    policy_engine.enforce_tool_call(auth, "search_documents")

    result = search_pages(
        query=query,
        persona_id=persona_id,
        top_k=_bounded_top_k(top_k),
        status_filter=status_filter,
        language=language,
    )
    result["tool_name"] = "search_documents"
    result["tool_policy"] = {
        "persona_id": persona_id,
        "decision": "allow",
        "reason": "persona is allowed to call read-only document search",
    }
    record_search_documents_state(
        tool_context=tool_context,
        query=query,
        persona_id=persona_id,
        result=result,
    )
    return result
