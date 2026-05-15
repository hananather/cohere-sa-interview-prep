"""Shared ADK state helpers for search tool results.

The live tool path and cached replay path must write the same session state.
Keeping that logic here prevents cache hits from becoming a second, weaker
trace format.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MAX_SOURCE_HISTORY = 48
MAX_AUDIT_HISTORY = 24


def record_search_documents_state(
    *,
    tool_context: Any | None,
    query: str,
    persona_id: str,
    result: dict[str, Any],
    cache_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a search result in ADK session state and return its audit."""

    sources_for_answer = _sources_for_answer(result)
    search_audit = _search_audit(query, persona_id, result, sources_for_answer)
    if cache_event is not None:
        search_audit["cache"] = dict(cache_event)
        result["tool_cache"] = dict(cache_event)
    result["search_audit"] = search_audit

    if tool_context is None:
        return search_audit

    actions = getattr(tool_context, "actions", None)
    if actions is not None:
        actions.skip_summarization = True

    state = tool_context.state
    state["last_search_query"] = query
    state["last_search_citations"] = result.get("citation_guide", [])
    state["last_search_persona_id"] = persona_id
    state["last_search_sources"] = sources_for_answer
    state["last_search_answerability"] = result.get("answerability", {})
    state["last_search_excluded_sources"] = search_audit["excluded_sources"]
    state["last_search_audit"] = search_audit
    state["search_history_sources"] = _merge_source_history(
        state.get("search_history_sources", []),
        sources_for_answer,
    )
    state["search_history_audits"] = _merge_audit_history(
        state.get("search_history_audits", []),
        search_audit,
    )
    return search_audit


def attach_cache_event_to_latest_search(
    *,
    tool_context: Any | None,
    result: dict[str, Any],
    cache_event: dict[str, Any],
) -> dict[str, Any]:
    """Attach cache metadata after a live tool call has already updated state."""

    result["tool_cache"] = dict(cache_event)
    audit = deepcopy(result.get("search_audit") or {})
    if audit:
        audit["cache"] = dict(cache_event)
        result["search_audit"] = audit

    if tool_context is None:
        return result

    state = tool_context.state
    state["last_tool_cache_event"] = dict(cache_event)
    if audit:
        state["last_search_audit"] = audit
        history = [item for item in state.get("search_history_audits", []) or [] if isinstance(item, dict)]
        if history:
            history[-1] = audit
            state["search_history_audits"] = history[-MAX_AUDIT_HISTORY:]
    return result


def _sources_for_answer(result: dict[str, Any]) -> list[dict[str, Any]]:
    answerability = result.get("answerability", {}) if isinstance(result.get("answerability", {}), dict) else {}
    reason = str(answerability.get("reason", "") or "")
    answerable = bool(answerability.get("answerable", True))
    if answerable or reason == "insufficient_authorized_evidence":
        return [source for source in result.get("authorized_sources", []) or [] if isinstance(source, dict)]
    return []


def _merge_source_history(existing: Any, new_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a small unique evidence set for final grounded generation."""

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in list(existing or []) + list(new_sources or []):
        if not isinstance(source, dict):
            continue
        key = str(source.get("chunk_id") or source.get("doc_id") or len(merged))
        if key in seen:
            continue
        seen.add(key)
        merged.append(source)
    return merged[-MAX_SOURCE_HISTORY:]


def _merge_audit_history(existing: Any, audit: dict[str, Any]) -> list[dict[str, Any]]:
    history = [item for item in list(existing or []) if isinstance(item, dict)]
    history.append(audit)
    return history[-MAX_AUDIT_HISTORY:]


def _search_audit(
    query: str,
    persona_id: str,
    result: dict[str, Any],
    sources_for_answer: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "query": query,
        "persona_id": persona_id,
        "index": result.get("index", ""),
        "collection": result.get("collection", ""),
        "embedding_backend": result.get("embedding_backend", ""),
        "rerank_backend": result.get("rerank_backend", ""),
        "allowed_access": list(result.get("allowed_access", []) or []),
        "filters_applied": dict(result.get("filters_applied", {}) or {}),
        "policy_decision": result.get("policy_decision", ""),
        "answerability": dict(result.get("answerability", {}) or {}),
        "authorized_sources": [_source_summary(source) for source in result.get("authorized_sources", []) or []],
        "sources_sent_to_answer": [_source_summary(source) for source in sources_for_answer],
        "excluded_sources": [_excluded_source_summary(source) for source in result.get("excluded_sources", []) or []],
    }


def _source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "citation_id": source.get("citation_id", ""),
        "chunk_id": source.get("chunk_id", ""),
        "doc_id": source.get("doc_id", ""),
        "title": source.get("title", ""),
        "section": source.get("section", ""),
        "page": source.get("page", ""),
        "status": source.get("status", ""),
        "version": source.get("version", ""),
        "effective_date": source.get("effective_date", ""),
        "access_level": source.get("access_level", ""),
        "language": source.get("language", ""),
        "source_type": source.get("source_type", ""),
        "source_format": source.get("source_format", ""),
        "normalized_format": source.get("normalized_format", ""),
        "normalization_method": source.get("normalization_method", ""),
        "canonical_url": source.get("canonical_url", ""),
        "source_url": source.get("source_url", ""),
        "retrieved_date": source.get("retrieved_date", ""),
        "source_organization": source.get("source_organization", ""),
        "owner": source.get("owner", ""),
        "source_docx_url": source.get("source_docx_url", ""),
        "source_pdf_url": source.get("source_pdf_url", ""),
        "provenance_note": source.get("provenance_note", ""),
        "source_pdf_path": source.get("source_pdf_path", ""),
        "manifest_path": source.get("manifest_path", ""),
        "page_image_sha256": source.get("page_image_sha256", ""),
        "vector_score": source.get("vector_score"),
        "rerank_score": source.get("rerank_score"),
    }


def _excluded_source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": source.get("doc_id", ""),
        "title": source.get("title", ""),
        "access_level": source.get("access_level", ""),
        "doc_family": source.get("doc_family", ""),
        "language": source.get("language", ""),
        "reason": source.get("reason", ""),
        "vector_score": source.get("vector_score"),
    }
