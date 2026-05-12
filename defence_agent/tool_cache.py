"""Session-scoped ADK tool-result cache.

This is cached replay, not exact invocation resume. ADK still owns the agent
loop. On retry, a matching deterministic tool call can be answered from ADK
session state instead of repeating retrieval, embedding, and reranking.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from time import time
from typing import Any

from defence_agent.auth.context import DEFAULT_PERSONA_ID, DEMO_USERS
from defence_agent.auth.policy import policy_engine
from defence_agent.config import get_settings
from defence_agent.retrieval.index_metadata import COLLECTION_PREFIX, INDEX_VERSION
from defence_agent.tool_state import attach_cache_event_to_latest_search, record_search_documents_state


CACHE_SCHEMA_VERSION = "search_tool_result_cache_v1"
SESSION_CACHE_STATE_KEY = "tool_result_cache"
SESSION_CACHE_STATS_KEY = "tool_cache_stats"
CACHEABLE_TOOLS = {"search_documents"}
MAX_TOOL_TOP_K = 24


def before_tool_cache_lookup(tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
    """ADK before-tool callback that returns a cached tool result on hit."""

    if not _cache_enabled():
        return None
    tool_name = _tool_name(tool)
    if tool_name not in CACHEABLE_TOOLS:
        return None

    cache_key, payload = cache_key_for_tool(tool_name, args, tool_context)
    entry = _cache_entries(tool_context.state).get(cache_key)
    if not _valid_entry(entry, payload):
        return None

    result = deepcopy(entry["result"])
    cache_event = _cache_event(
        hit=True,
        stored=False,
        key=cache_key,
        tool_name=tool_name,
        payload=payload,
        source="adk_session_state",
    )
    if tool_name == "search_documents":
        record_search_documents_state(
            tool_context=tool_context,
            query=payload["args"]["query"],
            persona_id=payload["persona_id"],
            result=result,
            cache_event=cache_event,
        )
    else:
        result["tool_cache"] = cache_event
    _increment_stats(tool_context.state, "hits")
    tool_context.state["last_tool_cache_event"] = cache_event
    return result


def after_tool_cache_store(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: dict[str, Any],
) -> dict[str, Any] | None:
    """ADK after-tool callback that stores successful deterministic outputs."""

    if not _cache_enabled():
        return None
    tool_name = _tool_name(tool)
    if tool_name not in CACHEABLE_TOOLS or not isinstance(tool_response, dict):
        return None
    if _response_has_error(tool_response) or not _tool_policy_allows(tool_response):
        return None

    cache_key, payload = cache_key_for_tool(tool_name, args, tool_context)
    cache_result = _cacheable_result(tool_response)
    cache = _cache_entries(tool_context.state)
    cache[cache_key] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_epoch": time(),
        "key_payload": payload,
        "result": cache_result,
    }
    tool_context.state[SESSION_CACHE_STATE_KEY] = _trim_cache(cache)

    cache_event = _cache_event(
        hit=False,
        stored=True,
        key=cache_key,
        tool_name=tool_name,
        payload=payload,
        source="live_tool_call",
    )
    result = _cacheable_result(tool_response)
    if tool_name == "search_documents":
        attach_cache_event_to_latest_search(
            tool_context=tool_context,
            result=result,
            cache_event=cache_event,
        )
    else:
        result["tool_cache"] = cache_event
    _increment_stats(tool_context.state, "stores")
    tool_context.state["last_tool_cache_event"] = cache_event
    return result


def cache_key_for_tool(tool_name: str, args: dict[str, Any], tool_context: Any) -> tuple[str, dict[str, Any]]:
    payload = cache_key_payload(tool_name, args, tool_context)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), payload


def cache_key_payload(tool_name: str, args: dict[str, Any], tool_context: Any) -> dict[str, Any]:
    persona_id = _persona_id(tool_context)
    allowed_access = list(policy_engine.acl_filter(DEMO_USERS[persona_id]).allowed_classifications)
    settings = get_settings()
    manifest_path = settings.corpus_dir / "manifest.yaml"
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "tool_name": tool_name,
        "persona_id": persona_id,
        "allowed_access": allowed_access,
        "args": _normalized_args(tool_name, args),
        "index": {
            "index_version": INDEX_VERSION,
            "collection": f"{COLLECTION_PREFIX}_{settings.cohere_embed_output_dimension}",
            "corpus_manifest_sha256": _sha256_file(manifest_path),
        },
        "cohere": {
            "embed_model": settings.cohere_embed_model,
            "embed_output_dimension": settings.cohere_embed_output_dimension,
            "rerank_model": settings.cohere_rerank_model,
        },
    }


def _cache_enabled() -> bool:
    return bool(get_settings().tool_cache_enabled)


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", tool) or "")


def _persona_id(tool_context: Any) -> str:
    state = getattr(tool_context, "state", {}) if tool_context is not None else {}
    requested = state.get("persona_id") or DEFAULT_PERSONA_ID
    return requested if requested in DEMO_USERS else DEFAULT_PERSONA_ID


def _normalized_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name != "search_documents":
        return _jsonable(args)
    return {
        "query": str(args.get("query", "") or "").strip(),
        "top_k": _bounded_top_k(args.get("top_k", 8)),
        "status_filter": _normalize_status_filter(str(args.get("status_filter", "approved") or "approved")),
        "language": _normalize_language_filter(str(args.get("language", "any") or "any")),
    }


def _bounded_top_k(top_k: Any) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        value = 8
    return min(max(value, 1), MAX_TOOL_TOP_K)


def _normalize_status_filter(status_filter: str) -> str:
    normalized = status_filter.strip().lower()
    aliases = {
        "current": "approved",
        "current_approved": "approved",
        "old": "superseded",
        "older": "superseded",
        "all": "any",
        "*": "any",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"approved", "draft", "superseded", "any"} else "approved"


def _normalize_language_filter(language: str) -> str:
    normalized = language.strip().lower()
    aliases = {
        "english": "en",
        "eng": "en",
        "french": "fr",
        "français": "fr",
        "francais": "fr",
        "all": "any",
        "*": "any",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"en", "fr", "any"} else "any"


def _sha256_file(path: Any) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return "missing"


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _cache_entries(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    existing = state.get(SESSION_CACHE_STATE_KEY, {})
    if not isinstance(existing, dict):
        return {}
    return {str(key): value for key, value in existing.items() if isinstance(value, dict)}


def _valid_entry(entry: Any, payload: dict[str, Any]) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("schema_version") == CACHE_SCHEMA_VERSION
        and entry.get("key_payload") == payload
        and isinstance(entry.get("result"), dict)
    )


def _response_has_error(tool_response: dict[str, Any]) -> bool:
    if "error" in tool_response:
        return True
    return str(tool_response.get("status", "")).lower() == "error"


def _tool_policy_allows(tool_response: dict[str, Any]) -> bool:
    policy = tool_response.get("tool_policy", {})
    return not isinstance(policy, dict) or policy.get("decision", "allow") == "allow"


def _cacheable_result(tool_response: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(tool_response)
    result.pop("tool_cache", None)
    audit = result.get("search_audit")
    if isinstance(audit, dict):
        audit.pop("cache", None)
        _strip_excluded_source_text(audit.get("excluded_sources", []))
    _strip_excluded_source_text(result.get("excluded_sources", []))
    return result


def _strip_excluded_source_text(sources: Any) -> None:
    for source in sources or []:
        if isinstance(source, dict):
            source.pop("text", None)


def _trim_cache(cache: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    max_entries = get_settings().tool_cache_max_entries
    ordered = sorted(
        cache.items(),
        key=lambda item: float(item[1].get("created_at_epoch", 0.0) or 0.0),
    )
    return dict(ordered[-max_entries:])


def _cache_event(
    *,
    hit: bool,
    stored: bool,
    key: str,
    tool_name: str,
    payload: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "scope": "session",
        "hit": hit,
        "stored": stored,
        "source": source,
        "tool_name": tool_name,
        "cache_key": key,
        "cache_key_short": key[:12],
        "persona_id": payload["persona_id"],
        "normalized_args": payload["args"],
    }


def _increment_stats(state: dict[str, Any], field: str) -> None:
    stats = dict(state.get(SESSION_CACHE_STATS_KEY, {}) or {})
    stats[field] = int(stats.get(field, 0) or 0) + 1
    state[SESSION_CACHE_STATS_KEY] = stats
