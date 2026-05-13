"""ADK runner and lightweight persistent session helpers.

This file is intentionally small. ADK owns the agent loop and the session
history. We only choose the session store, attach persona state, and return a
plain result object that is easy to inspect in a demo or test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types

from .agent import root_agent
from .auth.context import DEFAULT_PERSONA_ID, DEMO_USERS
from .config import get_settings
from .grounding import GroundedAnswer, finalize_answer


APP_NAME = "defence_agent"
logger = logging.getLogger(__name__)


class SessionPersonaMismatchError(PermissionError):
    """Raised when a persisted ADK session is reused with a different persona."""


@dataclass(frozen=True)
class AgentTurnResult:
    """Readable result from one ADK turn."""

    session_id: str
    user_id: str
    persona_id: str
    answer: str
    raw_answer: str
    events_seen: int
    tool_calls: list[str] = field(default_factory=list)
    tool_responses: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    citation_mode: str = "none"
    citation_validation: dict[str, Any] = field(default_factory=dict)
    grounded_model: str = ""
    documents_sent_to_model: int = 0
    retrieval_status: str = ""
    answer_audit: dict[str, Any] = field(default_factory=dict)


def default_session_db_path() -> Path:
    """Return the local SQLite database path for ADK sessions."""

    path = get_settings().data_dir / "sessions" / "adk_sessions.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sqlite_session_url(path: Path | None = None) -> str:
    """Build the async SQLAlchemy URL expected by ADK's DatabaseSessionService."""

    db_path = (path or default_session_db_path()).resolve()
    return f"sqlite+aiosqlite:///{db_path}"


def create_session_service(*, persistent: bool = True) -> DatabaseSessionService | InMemorySessionService:
    """Create the ADK session service.

    Persistent mode stores sessions in local SQLite. In-memory mode is useful
    for fast tests that do not need to inspect the database file.
    """

    if persistent:
        return DatabaseSessionService(db_url=sqlite_session_url())
    return InMemorySessionService()


async def ensure_session(
    session_service: DatabaseSessionService | InMemorySessionService,
    *,
    user_id: str,
    persona_id: str,
    session_id: str | None = None,
) -> str:
    """Create a session if it does not already exist."""

    resolved_session_id = session_id or f"s_{uuid4().hex}"
    existing = await _maybe_await(
        session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=resolved_session_id,
        )
    )
    if existing is None:
        await _maybe_await(
            session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=resolved_session_id,
                state={"persona_id": persona_id},
            )
        )
    else:
        state = getattr(existing, "state", {}) or {}
        stored_persona_id = str(state.get("persona_id", "") or "")
        if stored_persona_id and stored_persona_id != persona_id:
            raise SessionPersonaMismatchError(
                f"Session {resolved_session_id} belongs to {stored_persona_id}, not {persona_id}."
            )
        if not stored_persona_id:
            raise SessionPersonaMismatchError(
                f"Session {resolved_session_id} has no persona binding and cannot be reused safely."
            )
    return resolved_session_id


async def run_turn(
    query: str,
    *,
    persona_id: str = DEFAULT_PERSONA_ID,
    user_id: str | None = None,
    session_id: str | None = None,
    session_service: DatabaseSessionService | InMemorySessionService | None = None,
    target_answer_language: str = "auto",
) -> AgentTurnResult:
    """Run one ADK agent turn and return the final answer.

    Use the same ``session_id`` for follow-up questions. Use a different
    ``user_id`` or ``persona_id`` for persona-isolated conversations.
    """

    if persona_id not in DEMO_USERS:
        raise ValueError(f"Unknown persona_id: {persona_id}")

    resolved_user_id = user_id or persona_id
    service = session_service or create_session_service(persistent=True)
    resolved_session_id = await ensure_session(
        service,
        user_id=resolved_user_id,
        persona_id=persona_id,
        session_id=session_id,
    )
    session = await _maybe_await(
        service.get_session(
            app_name=APP_NAME,
            user_id=resolved_user_id,
            session_id=resolved_session_id,
        )
    )
    state = getattr(session, "state", {}) or {}
    prior_answer = str(state.get("last_grounded_answer", "")).strip()
    existing_source_keys = _source_keys(state.get("search_history_sources", []))
    existing_audit_count = _audit_count(state.get("search_history_audits", []))
    prior_answer_audit = state.get("last_answer_audit")
    if isinstance(prior_answer_audit, dict) and _is_audit_follow_up(query):
        return await _audit_follow_up_result(
            query=query,
            session_service=service,
            session_id=resolved_session_id,
            user_id=resolved_user_id,
            persona_id=persona_id,
            prior_audit=prior_answer_audit,
        )

    runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=service)
    message = types.Content(role="user", parts=[types.Part.from_text(text=_message_text(query, prior_answer))])
    event_count = 0
    tool_calls: list[str] = []
    tool_call_records: list[dict[str, Any]] = []
    tool_responses: list[str] = []
    retrieval_status = ""

    logger.info(
        "agent_turn_intent",
        extra={"persona_id": persona_id, "user_id": resolved_user_id, "session_id": resolved_session_id},
    )
    async for event in runner.run_async(
        user_id=resolved_user_id,
        session_id=resolved_session_id,
        new_message=message,
    ):
        event_count += 1
        tool_calls.extend(_function_call_names(event))
        tool_call_records.extend(_function_call_records(event))
        tool_responses.extend(_function_response_names(event))
        if event.is_final_response():
            retrieval_status = _event_text(event)

    final_state = await _session_state(service, user_id=resolved_user_id, session_id=resolved_session_id)
    turn_audits = _turn_audits(final_state, existing_audit_count)
    turn_sources = _turn_sources_from_audits(final_state, turn_audits) or _turn_sources(
        final_state,
        existing_source_keys,
    )
    logger.info(
        "agent_retrieval_outcome",
        extra={
            "persona_id": persona_id,
            "session_id": resolved_session_id,
            "tool_calls": tool_calls,
            "tool_responses": tool_responses,
            "authorized_source_count": len(turn_sources),
        },
    )
    grounded = _safe_finalize_answer(
        query=query,
        sources=turn_sources,
        prior_answer=prior_answer,
        fallback_answer=_fallback_answer(retrieval_status),
        target_answer_language=target_answer_language,
    )
    answer_audit = _answer_audit(
        query=query,
        session_id=resolved_session_id,
        user_id=resolved_user_id,
        persona_id=persona_id,
        tool_calls=tool_calls,
        tool_call_records=tool_call_records,
        tool_responses=tool_responses,
        retrieval_status=retrieval_status,
        retrieval_audits=turn_audits,
        sources_sent_to_answer=turn_sources,
        grounded=grounded,
        target_answer_language=target_answer_language,
    )
    await _update_session_grounding(
        service,
        user_id=resolved_user_id,
        session_id=resolved_session_id,
        grounded=grounded,
        answer_audit=answer_audit,
    )

    return AgentTurnResult(
        session_id=resolved_session_id,
        user_id=resolved_user_id,
        persona_id=persona_id,
        answer=grounded.answer,
        raw_answer=grounded.raw_answer,
        events_seen=event_count,
        tool_calls=tool_calls,
        tool_responses=tool_responses,
        citations=grounded.citations,
        citation_mode=grounded.citation_mode,
        citation_validation=grounded.citation_validation,
        grounded_model=grounded.model,
        documents_sent_to_model=grounded.documents_sent,
        retrieval_status=retrieval_status,
        answer_audit=answer_audit,
    )


def _is_audit_follow_up(query: str) -> bool:
    """Detect source-trace follow-ups that should use prior audit metadata."""

    text = query.lower()
    trace_terms = {
        "access level",
        "citation",
        "cite",
        "doc id",
        "document",
        "evidence",
        "page",
        "source",
        "support",
        "supports",
        "why could",
        "why couldn't",
    }
    direct_trace_phrases = {
        "what access level",
        "what document",
        "what page",
        "what source",
        "which access level",
        "which document",
        "which page",
        "which source",
    }
    reference_terms = {"that", "previous", "last", "above", "answer", "it"}
    return any(term in text for term in trace_terms) and (
        any(phrase in text for phrase in direct_trace_phrases)
        or any(term in text.split() for term in reference_terms)
    )


async def _audit_follow_up_result(
    *,
    query: str,
    session_service: DatabaseSessionService | InMemorySessionService,
    session_id: str,
    user_id: str,
    persona_id: str,
    prior_audit: dict[str, Any],
) -> AgentTurnResult:
    supporting_sources = _prior_supporting_sources(prior_audit)
    excluded_sources = _restricted_exclusion_summary(prior_audit.get("retrieval", {}).get("excluded_sources", []) or [])
    grounded = _audit_follow_up_grounded(supporting_sources, excluded_sources)
    answer_audit = _audit_lookup_audit(
        query=query,
        session_id=session_id,
        user_id=user_id,
        persona_id=persona_id,
        prior_audit=prior_audit,
        supporting_sources=supporting_sources,
        excluded_sources=excluded_sources,
        grounded=grounded,
    )
    await _update_session_grounding(
        session_service,
        user_id=user_id,
        session_id=session_id,
        grounded=grounded,
        answer_audit=answer_audit,
    )
    return AgentTurnResult(
        session_id=session_id,
        user_id=user_id,
        persona_id=persona_id,
        answer=grounded.answer,
        raw_answer=grounded.raw_answer,
        events_seen=0,
        tool_calls=["answer_audit_lookup"],
        tool_responses=["answer_audit_lookup"],
        citations=grounded.citations,
        citation_mode=grounded.citation_mode,
        citation_validation=grounded.citation_validation,
        grounded_model=grounded.model,
        documents_sent_to_model=grounded.documents_sent,
        retrieval_status="audit_lookup_complete",
        answer_audit=answer_audit,
    )


def _prior_supporting_sources(prior_audit: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for citation in prior_audit.get("citations", []) or []:
        if not isinstance(citation, dict):
            continue
        for source in citation.get("sources", []) or []:
            if isinstance(source, dict):
                sources.append(source)
    if not sources:
        sources = [
            source
            for source in prior_audit.get("retrieval", {}).get("sources_sent_to_answer", []) or []
            if isinstance(source, dict)
        ]
    return _dedupe_summaries(sources)


def _audit_follow_up_grounded(
    supporting_sources: list[dict[str, Any]],
    excluded_sources: list[dict[str, Any]],
) -> GroundedAnswer:
    if supporting_sources:
        lines = ["The previous answer was supported by:"]
        citations: list[dict[str, Any]] = []
        for index, source in enumerate(supporting_sources, start=1):
            label = f"C{index}"
            page = source.get("page", "")
            access_level = source.get("access_level", "")
            doc_id = source.get("doc_id", "")
            title = source.get("title", "")
            source_id = source.get("source_id") or source.get("chunk_id") or doc_id
            lines.append(
                f"- {doc_id} page {page}, access level {access_level}, source id {source_id} ({title}) [{label}]"
            )
            citations.append(
                {
                    "type": "answer_audit_lookup",
                    "start": None,
                    "end": None,
                    "text": str(doc_id),
                    "sources": [
                        {
                            **source,
                            "source_id": source_id,
                            "label": label,
                        }
                    ],
                }
            )
        answer = "\n".join(lines)
        return GroundedAnswer(
            answer=answer,
            raw_answer=answer,
            citations=citations,
            citation_mode="answer_audit_lookup",
            citation_validation={"passed": True, "errors": [], "citation_count": len(citations)},
            documents_sent=0,
            document_ids=[str(source.get("chunk_id") or source.get("doc_id", "")) for source in supporting_sources],
            model="answer_audit_lookup",
        )

    if excluded_sources:
        lines = [
            "The previous turn did not send authorized evidence to the final answer model.",
            "Denied-source metadata was present, but excluded source text was not exposed:",
        ]
        for source in excluded_sources:
            lines.append(
                f"- {source.get('count', 0)} source(s), access level {source.get('access_level', '')}, reason {source.get('reason', '')}"
            )
        answer = "\n".join(lines)
        return GroundedAnswer(
            answer=answer,
            raw_answer=answer,
            citation_mode="answer_audit_lookup",
            citation_validation={"passed": True, "errors": [], "citation_count": 0},
            documents_sent=0,
            document_ids=[],
            model="answer_audit_lookup",
        )

    answer = "The previous turn did not contain source metadata that can answer this audit follow-up."
    return GroundedAnswer(
        answer=answer,
        raw_answer=answer,
        citation_mode="answer_audit_lookup",
        citation_validation={"passed": False, "errors": ["no_prior_source_metadata"], "citation_count": 0},
        documents_sent=0,
        document_ids=[],
        model="answer_audit_lookup",
    )


def _audit_lookup_audit(
    *,
    query: str,
    session_id: str,
    user_id: str,
    persona_id: str,
    prior_audit: dict[str, Any],
    supporting_sources: list[dict[str, Any]],
    excluded_sources: list[dict[str, Any]],
    grounded: GroundedAnswer,
) -> dict[str, Any]:
    return {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "persona_id": persona_id,
        "tool_calls": ["answer_audit_lookup"],
        "tool_call_records": [{"tool_name": "answer_audit_lookup", "args": {"source": "last_answer_audit"}}],
        "tool_responses": ["answer_audit_lookup"],
        "retrieval_status": "audit_lookup_complete",
        "audit_lookup": {
            "source": "last_answer_audit",
            "previous_query": prior_audit.get("query", ""),
            "supporting_sources": supporting_sources,
            "excluded_sources": excluded_sources,
        },
        "retrieval": {
            "search_count": 0,
            "allowed_access": prior_audit.get("retrieval", {}).get("allowed_access", []),
            "filters_applied": [],
            "policy_decisions": ["audit_lookup"],
            "answerability": [],
            "authorized_sources": supporting_sources,
            "sources_sent_to_answer": supporting_sources,
            "excluded_sources": excluded_sources,
        },
        "generation": {
            "model": grounded.model,
            "citation_mode": grounded.citation_mode,
            "citation_resolution": grounded.citation_validation,
            "citation_quality": _citation_quality_summary(grounded.citation_validation),
            "document_count": grounded.documents_sent,
            "cohere_document_ids": grounded.document_ids,
            "target_answer_language": "auto",
        },
        "citations": grounded.citations,
    }


def _event_text(event: Any) -> str:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(str(getattr(part, "text", "") or "") for part in parts).strip()


def _message_text(query: str, prior_answer: str) -> str:
    if not prior_answer:
        return query
    return (
        "Session context from the previous assistant answer:\n"
        f"{prior_answer[:1800]}\n\n"
        "User follow-up:\n"
        f"{query}"
    )


def _function_call_names(event: Any) -> list[str]:
    try:
        calls = event.get_function_calls()
    except Exception:
        return []
    return [str(getattr(call, "name", "")) for call in calls if getattr(call, "name", "")]


def _function_call_records(event: Any) -> list[dict[str, Any]]:
    try:
        calls = event.get_function_calls()
    except Exception:
        return []
    records: list[dict[str, Any]] = []
    for call in calls:
        name = str(getattr(call, "name", "") or "")
        if not name:
            continue
        records.append(
            {
                "tool_name": name,
                "args": _safe_tool_args(getattr(call, "args", None)),
            }
        )
    return records


def _function_response_names(event: Any) -> list[str]:
    try:
        responses = event.get_function_responses()
    except Exception:
        return []
    return [str(getattr(response, "name", "")) for response in responses if getattr(response, "name", "")]


def _safe_tool_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return {str(key): _json_safe_value(value) for key, value in raw_args.items()}
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {"raw": raw_args[:500]}
        if isinstance(parsed, dict):
            return {str(key): _json_safe_value(value) for key, value in parsed.items()}
    return {}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _session_state(
    session_service: DatabaseSessionService | InMemorySessionService,
    *,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    session = await _maybe_await(
        session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
    )
    return dict(getattr(session, "state", {}) or {})


def _audit_count(audits: Any) -> int:
    return len([audit for audit in audits or [] if isinstance(audit, dict)])


def _source_keys(sources: Any) -> set[str]:
    keys: set[str] = set()
    for source in sources or []:
        if isinstance(source, dict):
            keys.add(str(source.get("chunk_id") or source.get("doc_id") or ""))
    return {key for key in keys if key}


def _source_key(source: dict[str, Any]) -> str:
    return str(source.get("chunk_id") or source.get("doc_id") or "")


def _turn_audits(state: dict[str, Any], existing_audit_count: int) -> list[dict[str, Any]]:
    history = [audit for audit in state.get("search_history_audits", []) or [] if isinstance(audit, dict)]
    current_turn = history[existing_audit_count:]
    if current_turn:
        return current_turn
    last_audit = state.get("last_search_audit")
    return [last_audit] if isinstance(last_audit, dict) else []


def _turn_sources_from_audits(state: dict[str, Any], audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: set[str] = set()
    for audit in audits:
        for source in audit.get("sources_sent_to_answer", []) or []:
            if isinstance(source, dict):
                key = _source_key(source)
                if key:
                    keys.add(key)
    if not keys:
        return []

    candidates = list(state.get("search_history_sources", []) or []) + list(state.get("last_search_sources", []) or [])
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in candidates:
        if not isinstance(source, dict):
            continue
        key = _source_key(source)
        if key in keys and key not in seen:
            selected.append(source)
            seen.add(key)
    return selected


def _turn_sources(state: dict[str, Any], existing_source_keys: set[str]) -> list[dict[str, Any]]:
    history = state.get("search_history_sources", []) or []
    current_turn = [
        source
        for source in history
        if isinstance(source, dict)
        and str(source.get("chunk_id") or source.get("doc_id") or "") not in existing_source_keys
    ]
    if current_turn:
        return current_turn
    return [source for source in state.get("last_search_sources", []) or [] if isinstance(source, dict)]


def _answer_audit(
    *,
    query: str,
    session_id: str,
    user_id: str,
    persona_id: str,
    tool_calls: list[str],
    tool_responses: list[str],
    retrieval_status: str,
    retrieval_audits: list[dict[str, Any]],
    sources_sent_to_answer: list[dict[str, Any]],
    grounded: GroundedAnswer,
    tool_call_records: list[dict[str, Any]] | None = None,
    target_answer_language: str = "auto",
) -> dict[str, Any]:
    source_summaries = [_source_summary(source) for source in sources_sent_to_answer]
    lookup = _source_lookup(source_summaries, retrieval_audits)
    return {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "persona_id": persona_id,
        "tool_calls": tool_calls,
        "tool_call_records": list(tool_call_records or []),
        "tool_responses": tool_responses,
        "retrieval_status": retrieval_status,
        "retrieval": {
            "search_count": len(retrieval_audits),
            "searches": _search_summaries(retrieval_audits),
            "search_queries": _search_queries(retrieval_audits),
            "tool_cache": _tool_cache_summary(retrieval_audits),
            "allowed_access": _unique_values(retrieval_audits, "allowed_access"),
            "filters_applied": [audit.get("filters_applied", {}) for audit in retrieval_audits],
            "policy_decisions": [audit.get("policy_decision", "") for audit in retrieval_audits],
            "answerability": [audit.get("answerability", {}) for audit in retrieval_audits],
            "authorized_sources": _dedupe_summaries(
                source
                for audit in retrieval_audits
                for source in audit.get("authorized_sources", []) or []
                if isinstance(source, dict)
            ),
            "sources_sent_to_answer": source_summaries,
            "excluded_sources": _dedupe_summaries(
                source
                for audit in retrieval_audits
                for source in audit.get("excluded_sources", []) or []
                if isinstance(source, dict)
            ),
        },
        "generation": {
            "model": grounded.model,
            "citation_mode": grounded.citation_mode,
            "citation_resolution": grounded.citation_validation,
            "citation_quality": _citation_quality_summary(grounded.citation_validation),
            "document_count": grounded.documents_sent,
            "cohere_document_ids": grounded.document_ids,
            "target_answer_language": target_answer_language,
        },
        "citations": [_citation_summary(citation, lookup) for citation in grounded.citations],
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


def _source_lookup(
    source_summaries: list[dict[str, Any]],
    retrieval_audits: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    all_sources = list(source_summaries)
    for audit in retrieval_audits:
        all_sources.extend(source for source in audit.get("authorized_sources", []) or [] if isinstance(source, dict))
    for source in all_sources:
        for key in {
            str(source.get("chunk_id", "")),
            str(source.get("doc_id", "")),
            str(source.get("citation_id", "")),
        }:
            if key:
                lookup.setdefault(key, source)
    return lookup


def _citation_summary(citation: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    raw_sources = citation.get("sources", []) or []
    if not raw_sources and citation.get("doc_id"):
        raw_sources = [
            {
                "source_id": citation.get("chunk_id") or citation.get("doc_id", ""),
                "label": citation.get("label", ""),
                "doc_id": citation.get("doc_id", ""),
                "title": citation.get("title", ""),
                "page": citation.get("page", ""),
                "language": citation.get("language", ""),
                "access_level": citation.get("access_level", ""),
                "status": citation.get("status", ""),
                "version": citation.get("version", ""),
                "chunk_id": citation.get("chunk_id", ""),
            }
        ]
    for source in raw_sources:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("source_id", ""))
        resolved = lookup.get(source_id) or lookup.get(str(source.get("chunk_id", ""))) or lookup.get(
            str(source.get("doc_id", ""))
        ) or lookup.get(str(source.get("label", ""))) or {}
        sources.append(
            {
                "source_id": source_id,
                "label": source.get("label", ""),
                "doc_id": source.get("doc_id") or resolved.get("doc_id", ""),
                "title": source.get("title") or resolved.get("title", ""),
                "page": source.get("page") or resolved.get("page", ""),
                "language": source.get("language") or resolved.get("language", ""),
                "access_level": source.get("access_level") or resolved.get("access_level", ""),
                "source_type": source.get("source_type") or resolved.get("source_type", ""),
                "source_format": source.get("source_format") or resolved.get("source_format", ""),
                "normalized_format": source.get("normalized_format") or resolved.get("normalized_format", ""),
                "normalization_method": resolved.get("normalization_method", ""),
                "canonical_url": source.get("canonical_url") or resolved.get("canonical_url", ""),
                "source_url": source.get("source_url") or resolved.get("source_url", ""),
                "source_pdf_url": source.get("source_pdf_url") or resolved.get("source_pdf_url", ""),
                "source_docx_url": source.get("source_docx_url") or resolved.get("source_docx_url", ""),
                "retrieved_date": source.get("retrieved_date") or resolved.get("retrieved_date", ""),
                "source_organization": source.get("source_organization") or resolved.get("source_organization", ""),
                "provenance_note": source.get("provenance_note") or resolved.get("provenance_note", ""),
                "status": source.get("status") or resolved.get("status", ""),
                "version": source.get("version") or resolved.get("version", ""),
                "chunk_id": source.get("chunk_id") or resolved.get("chunk_id", ""),
                "source_pdf_path": resolved.get("source_pdf_path", ""),
                "manifest_path": resolved.get("manifest_path", ""),
                "page_image_sha256": resolved.get("page_image_sha256", ""),
                "vector_score": resolved.get("vector_score"),
                "rerank_score": resolved.get("rerank_score"),
            }
        )
    return {
        "type": citation.get("type", ""),
        "start": citation.get("start"),
        "end": citation.get("end"),
        "text": citation.get("text", ""),
        "sources": sources,
    }


def _citation_quality_summary(validation: dict[str, Any]) -> dict[str, Any]:
    coverage = validation.get("coverage", {}) if isinstance(validation, dict) else {}
    claim_count = int(coverage.get("claim_count", 0) or 0)
    covered_claim_count = int(coverage.get("covered_claim_count", 0) or 0)
    recall_proxy = round(covered_claim_count / claim_count, 4) if claim_count else None
    return {
        "citation_recall_proxy": recall_proxy,
        "claim_count": claim_count,
        "covered_claim_count": covered_claim_count,
        "uncited_claim_count": int(coverage.get("uncited_claim_count", 0) or 0),
        "citation_precision": "manual_or_llm_judge_required",
        "note": (
            "Span coverage and source resolution are automated checks. "
            "They do not prove that each citation supports its claim."
        ),
    }


def _search_queries(audits: list[dict[str, Any]]) -> list[str]:
    queries: list[str] = []
    for audit in audits:
        query = str(audit.get("query", "")).strip()
        if query:
            queries.append(query)
    return queries


def _search_summaries(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, audit in enumerate(audits, start=1):
        if not isinstance(audit, dict):
            continue
        answerability = audit.get("answerability", {}) if isinstance(audit.get("answerability"), dict) else {}
        evidence_quality = (
            answerability.get("evidence_quality", {}) if isinstance(answerability.get("evidence_quality"), dict) else {}
        )
        summaries.append(
            {
                "call_index": index,
                "query": audit.get("query", ""),
                "filters_applied": dict(audit.get("filters_applied", {}) or {}),
                "policy_decision": audit.get("policy_decision", ""),
                "answerable": answerability.get("answerable"),
                "answerability_reason": answerability.get("reason", ""),
                "authorized_source_count": len(audit.get("authorized_sources", []) or []),
                "sources_sent_to_answer_count": len(audit.get("sources_sent_to_answer", []) or []),
                "excluded_source_count": len(audit.get("excluded_sources", []) or []),
                "embedding_backend": audit.get("embedding_backend", ""),
                "rerank_backend": audit.get("rerank_backend", ""),
                "collection": audit.get("collection", ""),
                "top_authorized_vector_score": evidence_quality.get("top_authorized_vector_score"),
                "top_authorized_rerank_score": evidence_quality.get("top_authorized_rerank_score"),
            }
        )
    return summaries


def _tool_cache_summary(audits: list[dict[str, Any]]) -> dict[str, Any]:
    events = [audit.get("cache") for audit in audits if isinstance(audit.get("cache"), dict)]
    return {
        "events": events,
        "hit_count": sum(1 for event in events if event.get("hit")),
        "store_count": sum(1 for event in events if event.get("stored")),
    }


def _unique_values(audits: list[dict[str, Any]], key: str) -> list[Any]:
    seen: list[Any] = []
    for audit in audits:
        value = audit.get(key, [])
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item not in seen:
                seen.append(item)
    return seen


def _dedupe_summaries(sources: Any) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        key = _source_key(source) or str(source.get("title", "")) or str(len(deduped))
        if key in seen:
            continue
        seen.add(key)
        clean = {field: value for field, value in source.items() if field != "text"}
        deduped.append(clean)
    return deduped


def _restricted_exclusion_summary(sources: Any) -> list[dict[str, Any]]:
    summary: dict[tuple[str, str], int] = {}
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        access_level = str(source.get("access_level", "restricted") or "restricted")
        reason = str(source.get("reason", "access_denied") or "access_denied")
        summary[(access_level, reason)] = summary.get((access_level, reason), 0) + 1
    return [
        {"access_level": access_level, "reason": reason, "count": count}
        for (access_level, reason), count in sorted(summary.items())
    ]


def _safe_finalize_answer(
    *,
    query: str,
    sources: list[dict[str, Any]],
    prior_answer: str,
    fallback_answer: str,
    target_answer_language: str,
) -> GroundedAnswer:
    try:
        return finalize_answer(
            query=query,
            sources=sources,
            prior_answer=prior_answer,
            fallback_answer=fallback_answer,
            target_answer_language=target_answer_language,
        )
    except Exception as exc:
        return GroundedAnswer(
            answer=fallback_answer,
            raw_answer=fallback_answer,
            citation_mode="grounding_error",
            citation_validation={"passed": False, "errors": [f"{type(exc).__name__}: {exc}"]},
            documents_sent=len(sources),
            model=get_settings().cohere_chat_model,
        )


def _fallback_answer(retrieval_status: str) -> str:
    status = retrieval_status.strip()
    if status and (
        "missing" in status.lower()
        or "secret" in status.lower()
        or "unauthorized" in status.lower()
        or "ambiguous" in status.lower()
    ):
        return status
    return "I do not have enough authorized evidence to answer."


async def _update_session_grounding(
    session_service: DatabaseSessionService | InMemorySessionService,
    *,
    user_id: str,
    session_id: str,
    grounded: GroundedAnswer,
    answer_audit: dict[str, Any],
) -> None:
    session = await _maybe_await(
        session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
    )
    if session is None:
        return
    event = Event(
        author=APP_NAME,
        invocation_id=f"grounding_{uuid4().hex}",
        actions=EventActions(
            state_delta={
                "last_grounded_answer": grounded.answer,
                "last_citation_mode": grounded.citation_mode,
                "last_citation_validation": grounded.citation_validation,
                "last_answer_audit": answer_audit,
            }
        ),
    )
    await _maybe_await(session_service.append_event(session=session, event=event))
