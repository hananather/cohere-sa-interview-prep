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
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types

from .auth.context import DEFAULT_PERSONA_ID, DEMO_USERS
from .config import get_settings
from .critic import NEEDS_HUMAN_REVIEW, NEEDS_REVISION, review_answer
from .grounding import GroundedAnswer, finalize_answer
from .index import search_pages
from .routing import SIMPLE_RAG, RouteDecision, choose_route
from .tool_state import record_search_documents_state


APP_NAME = "defence_agent"
logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


DEFAULT_MAX_REVIEW_CYCLES = _env_int("DEFTECH_ADK_MAX_REVIEW_CYCLES", 2)
MAX_REVIEW_FEEDBACK_CHARS = _env_int("DEFTECH_ADK_REVIEW_FEEDBACK_CHARS", 3600)
MAX_REVIEW_CONTEXT_SOURCES = _env_int("DEFTECH_ADK_REVIEW_CONTEXT_SOURCES", 8)
CONTEXT_WINDOW_TOKEN_ESTIMATE = _env_int("DEFTECH_CONTEXT_WINDOW_TOKENS", 256000)
ESTIMATED_CHARS_PER_TOKEN = _env_int("DEFTECH_ESTIMATED_CHARS_PER_TOKEN", 4)
CONTEXT_ESTIMATE_OVERHEAD_TOKENS = _env_int("DEFTECH_CONTEXT_ESTIMATE_OVERHEAD_TOKENS", 450)


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
    thinking_blocks: list[dict[str, Any]] = field(default_factory=list)
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
    run_mode: str = "reviewed_agent",
    accuracy_priority: int = 4,
    latency_priority: int = 2,
    max_review_cycles: int | None = None,
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

    route = choose_route(
        requested_mode=run_mode,
        query=query,
        accuracy_priority=accuracy_priority,
        latency_priority=latency_priority,
    )
    if route.selected_mode == SIMPLE_RAG:
        return await _simple_rag_result(
            query=query,
            session_service=service,
            session_id=resolved_session_id,
            user_id=resolved_user_id,
            persona_id=persona_id,
            prior_answer=prior_answer,
            target_answer_language=target_answer_language,
            route=route,
        )

    from .agent import root_agent

    runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=service)
    message = types.Content(role="user", parts=[types.Part.from_text(text=_message_text(query, prior_answer))])
    event_count = 0
    tool_calls: list[str] = []
    tool_call_records: list[dict[str, Any]] = []
    tool_responses: list[str] = []
    retrieval_status = ""
    review_cycles: list[dict[str, Any]] = []
    review_cycle_limit = _review_cycle_limit(max_review_cycles)

    logger.info(
        "agent_turn_intent",
        extra={"persona_id": persona_id, "user_id": resolved_user_id, "session_id": resolved_session_id},
    )
    cycle_events, cycle_tool_calls, cycle_tool_call_records, cycle_tool_responses, cycle_status = (
        await _run_generator_agent_cycle(
            runner=runner,
            user_id=resolved_user_id,
            session_id=resolved_session_id,
            message=message,
        )
    )
    event_count += cycle_events
    tool_calls.extend(cycle_tool_calls)
    tool_call_records.extend(cycle_tool_call_records)
    tool_responses.extend(cycle_tool_responses)
    retrieval_status = cycle_status

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
        prior_answer=prior_answer,
        target_answer_language=target_answer_language,
    )
    cycle_index = 1
    while True:
        answer_audit["routing"] = route.as_audit()
        if not route.uses_reviewer:
            critic_report = _critic_skipped_report(route=route, citation_count=len(grounded.citations))
            answer_audit["reviewer"] = critic_report
            answer_audit["critic"] = critic_report
            break

        critic_report = await _critic_report(
            query=query,
            answer_audit=answer_audit,
            grounded=grounded,
            sources_sent_to_answer=turn_sources,
        )
        should_revise = _should_run_revision(
            critic_report,
            cycle_index=cycle_index,
            max_review_cycles=review_cycle_limit,
        )
        feedback_message = (
            _generator_feedback_message(
                query=query,
                grounded=grounded,
                answer_audit=answer_audit,
                critic_report=critic_report,
                cycle_index=cycle_index,
                max_review_cycles=review_cycle_limit,
            )
            if should_revise
            else ""
        )
        if not should_revise and _review_cycles_exhausted(critic_report, cycle_index, review_cycle_limit):
            critic_report = _max_cycles_human_review_report(critic_report, review_cycle_limit)
        answer_audit["reviewer"] = critic_report
        answer_audit["critic"] = critic_report
        review_cycles.append(
            _review_cycle_summary(
                cycle_index=cycle_index,
                max_review_cycles=review_cycle_limit,
                critic_report=critic_report,
                feedback_message=feedback_message,
                source_count=len(turn_sources),
                search_count=len(turn_audits),
            )
        )
        answer_audit["review_control"] = _review_control_summary(
            max_review_cycles=review_cycle_limit,
            review_cycles=review_cycles,
        )
        if not should_revise:
            break

        cycle_index += 1
        revision_message = types.Content(role="user", parts=[types.Part.from_text(text=feedback_message)])
        cycle_events, cycle_tool_calls, cycle_tool_call_records, cycle_tool_responses, cycle_status = (
            await _run_generator_agent_cycle(
                runner=runner,
                user_id=resolved_user_id,
                session_id=resolved_session_id,
                message=revision_message,
            )
        )
        event_count += cycle_events
        tool_calls.extend(cycle_tool_calls)
        tool_call_records.extend(cycle_tool_call_records)
        tool_responses.extend(cycle_tool_responses)
        retrieval_status = cycle_status
        final_state = await _session_state(service, user_id=resolved_user_id, session_id=resolved_session_id)
        turn_audits = _turn_audits(final_state, existing_audit_count)
        turn_sources = _turn_sources_from_audits(final_state, turn_audits) or _turn_sources(
            final_state,
            existing_source_keys,
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
            prior_answer=prior_answer,
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
        thinking_blocks=grounded.thinking_blocks,
        retrieval_status=retrieval_status,
        answer_audit=answer_audit,
    )


async def _simple_rag_result(
    *,
    query: str,
    session_service: DatabaseSessionService | InMemorySessionService,
    session_id: str,
    user_id: str,
    persona_id: str,
    prior_answer: str,
    target_answer_language: str,
    route: RouteDecision,
) -> AgentTurnResult:
    """Run one direct retrieval plus grounded answer without the ADK planner."""

    retrieval_language = target_answer_language if target_answer_language in {"en", "fr"} else "any"
    search_result = search_pages(
        query=query,
        persona_id=persona_id,
        top_k=8,
        status_filter="approved",
        language=retrieval_language,
    )
    search_audit = record_search_documents_state(
        tool_context=None,
        query=query,
        persona_id=persona_id,
        result=search_result,
    )
    sources_for_answer = _sources_for_answer(search_result)
    retrieval_status = "direct_retrieval_complete"
    grounded = _safe_finalize_answer(
        query=query,
        sources=sources_for_answer,
        prior_answer=prior_answer,
        fallback_answer=_fallback_answer(str(search_result.get("policy_decision", ""))),
        target_answer_language=target_answer_language,
    )
    tool_call_records = [
        {
            "tool_name": "search_documents",
            "args": {
                "query": query,
                "top_k": 8,
                "status_filter": "approved",
                "language": retrieval_language,
            },
        }
    ]
    answer_audit = _answer_audit(
        query=query,
        session_id=session_id,
        user_id=user_id,
        persona_id=persona_id,
        tool_calls=["search_documents"],
        tool_call_records=tool_call_records,
        tool_responses=["search_documents"],
        retrieval_status=retrieval_status,
        retrieval_audits=[search_audit],
        sources_sent_to_answer=sources_for_answer,
        grounded=grounded,
        prior_answer=prior_answer,
        target_answer_language=target_answer_language,
    )
    answer_audit["routing"] = route.as_audit()
    critic_report = _critic_skipped_report(route=route, citation_count=len(grounded.citations))
    answer_audit["reviewer"] = critic_report
    answer_audit["critic"] = critic_report
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
        tool_calls=["search_documents"],
        tool_responses=["search_documents"],
        citations=grounded.citations,
        citation_mode=grounded.citation_mode,
        citation_validation=grounded.citation_validation,
        grounded_model=grounded.model,
        documents_sent_to_model=grounded.documents_sent,
        thinking_blocks=grounded.thinking_blocks,
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
    critic_report = await _critic_report(
        query=query,
        answer_audit=answer_audit,
        grounded=grounded,
        sources_sent_to_answer=supporting_sources,
    )
    answer_audit["reviewer"] = critic_report
    answer_audit["critic"] = critic_report
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
        thinking_blocks=grounded.thinking_blocks,
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
            "content_blocks": grounded.content_blocks,
            "thinking_blocks": grounded.thinking_blocks,
            "thinking_block_count": len(grounded.thinking_blocks),
            "usage": grounded.usage,
            "billed_units": grounded.billed_units,
        },
        "context_budget": _context_budget_summary(
            query=query,
            prior_answer="",
            sources_sent_to_answer=supporting_sources,
            grounded=grounded,
            retrieval_audits=[],
            tool_call_records=[{"tool_name": "answer_audit_lookup", "args": {"source": "last_answer_audit"}}],
        ),
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


async def _run_generator_agent_cycle(
    *,
    runner: Runner,
    user_id: str,
    session_id: str,
    message: types.Content,
) -> tuple[int, list[str], list[dict[str, Any]], list[str], str]:
    event_count = 0
    tool_calls: list[str] = []
    tool_call_records: list[dict[str, Any]] = []
    tool_responses: list[str] = []
    retrieval_status = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        event_count += 1
        tool_calls.extend(_function_call_names(event))
        tool_call_records.extend(_function_call_records(event))
        tool_responses.extend(_function_response_names(event))
        if event.is_final_response():
            retrieval_status = _event_text(event)
    return event_count, tool_calls, tool_call_records, tool_responses, retrieval_status


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
    prior_answer: str = "",
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
            "retrieval_modes": _unique_values(retrieval_audits, "retrieval_mode"),
            "chunk_strategies": _unique_values(retrieval_audits, "chunk_strategy"),
            "retrieval_metrics": [audit.get("retrieval_metrics", {}) for audit in retrieval_audits],
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
            "content_blocks": grounded.content_blocks,
            "thinking_blocks": grounded.thinking_blocks,
            "thinking_block_count": len(grounded.thinking_blocks),
            "usage": grounded.usage,
            "billed_units": grounded.billed_units,
        },
        "context_budget": _context_budget_summary(
            query=query,
            prior_answer=prior_answer,
            sources_sent_to_answer=sources_sent_to_answer,
            grounded=grounded,
            retrieval_audits=retrieval_audits,
            tool_call_records=list(tool_call_records or []),
        ),
        "citations": [_citation_summary(citation, lookup) for citation in grounded.citations],
    }


def _context_budget_summary(
    *,
    query: str,
    prior_answer: str,
    sources_sent_to_answer: list[dict[str, Any]],
    grounded: GroundedAnswer,
    retrieval_audits: list[dict[str, Any]],
    tool_call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate the context footprint for the generation step.

    Cohere usage is provider truth when present. The other values are still
    useful in the demo because they show how much source text we are injecting.
    """

    source_text_chars = sum(len(_source_context_text(source)) for source in sources_sent_to_answer)
    source_metadata_chars = sum(
        len(json.dumps(_review_source_summary(source), ensure_ascii=False))
        for source in sources_sent_to_answer
        if isinstance(source, dict)
    )
    query_chars = len(str(query or ""))
    prior_answer_chars = min(len(str(prior_answer or "")), 1800)
    answer_chars = len(str(grounded.raw_answer or grounded.answer or ""))

    query_tokens = _estimate_tokens(query_chars)
    prior_answer_tokens = _estimate_tokens(prior_answer_chars)
    source_text_tokens = _estimate_tokens(source_text_chars)
    source_metadata_tokens = _estimate_tokens(source_metadata_chars)
    output_tokens = _estimate_tokens(answer_chars)
    prompt_tokens = (
        query_tokens
        + prior_answer_tokens
        + source_text_tokens
        + source_metadata_tokens
        + CONTEXT_ESTIMATE_OVERHEAD_TOKENS
    )
    total_tokens = prompt_tokens + output_tokens
    context_window = max(1, CONTEXT_WINDOW_TOKEN_ESTIMATE)
    provider_usage = dict(grounded.usage or {})
    provider_billed_units = dict(grounded.billed_units or {})

    return {
        "schema_version": "context_budget.v1",
        "method": f"estimate_{max(1, ESTIMATED_CHARS_PER_TOKEN)}_chars_per_token",
        "context_window_tokens": context_window,
        "context_window_used_pct": round((prompt_tokens / context_window) * 100, 2),
        "prompt_tokens_estimate": prompt_tokens,
        "output_tokens_estimate": output_tokens,
        "total_tokens_estimate": total_tokens,
        "query_tokens_estimate": query_tokens,
        "prior_answer_tokens_estimate": prior_answer_tokens,
        "source_text_tokens_estimate": source_text_tokens,
        "source_metadata_tokens_estimate": source_metadata_tokens,
        "system_overhead_tokens_estimate": CONTEXT_ESTIMATE_OVERHEAD_TOKENS,
        "source_text_chars": source_text_chars,
        "source_metadata_chars": source_metadata_chars,
        "document_count": grounded.documents_sent,
        "source_count": len(sources_sent_to_answer),
        "search_count": len(retrieval_audits),
        "tool_call_count": len(tool_call_records),
        "provider_usage_available": bool(provider_usage or provider_billed_units),
        "provider_usage": provider_usage,
        "provider_billed_units": provider_billed_units,
        "note": "Token counts are estimates unless provider usage is available.",
    }


def _source_context_text(source: dict[str, Any]) -> str:
    for field in ("text", "retrieval_chunk_text", "content", "page_text"):
        value = source.get(field)
        if value not in ("", None):
            return str(value)
    return ""


def _estimate_tokens(char_count: int) -> int:
    denominator = max(1, ESTIMATED_CHARS_PER_TOKEN)
    return max(0, int(round(max(0, char_count) / denominator)))


def _source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "citation_id": source.get("citation_id", ""),
        "chunk_id": source.get("chunk_id", ""),
        "parent_page_id": source.get("parent_page_id", ""),
        "retrieval_chunk_id": source.get("retrieval_chunk_id", ""),
        "chunk_strategy": source.get("chunk_strategy", ""),
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
        "bm25_score": source.get("bm25_score"),
        "pre_rerank_score": source.get("pre_rerank_score"),
        "rerank_score": source.get("rerank_score"),
        "retrieval_modes": source.get("retrieval_modes", []),
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
                "parent_page_id": source.get("parent_page_id") or resolved.get("parent_page_id", ""),
                "retrieval_chunk_id": source.get("retrieval_chunk_id") or resolved.get("retrieval_chunk_id", ""),
                "chunk_strategy": source.get("chunk_strategy") or resolved.get("chunk_strategy", ""),
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


def _sources_for_answer(search_result: dict[str, Any]) -> list[dict[str, Any]]:
    answerability = (
        search_result.get("answerability", {})
        if isinstance(search_result.get("answerability", {}), dict)
        else {}
    )
    reason = str(answerability.get("reason", "") or "")
    answerable = bool(answerability.get("answerable", True))
    if answerable or reason == "insufficient_authorized_evidence":
        return [source for source in search_result.get("authorized_sources", []) or [] if isinstance(source, dict)]
    return []


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


def _review_cycle_limit(value: int | None) -> int:
    requested = DEFAULT_MAX_REVIEW_CYCLES if value is None else value
    try:
        parsed = int(requested)
    except (TypeError, ValueError):
        parsed = 1
    return min(3, max(1, parsed))


def _should_run_revision(
    critic_report: dict[str, Any],
    *,
    cycle_index: int,
    max_review_cycles: int,
) -> bool:
    return (
        str(critic_report.get("status", "") or "") == NEEDS_REVISION
        and cycle_index < max_review_cycles
        and bool(str(critic_report.get("generator_feedback", "") or "").strip())
    )


def _review_cycles_exhausted(
    critic_report: dict[str, Any],
    cycle_index: int,
    max_review_cycles: int,
) -> bool:
    return str(critic_report.get("status", "") or "") == NEEDS_REVISION and cycle_index >= max_review_cycles


def _max_cycles_human_review_report(critic_report: dict[str, Any], max_review_cycles: int) -> dict[str, Any]:
    report = dict(critic_report)
    report["status"] = NEEDS_HUMAN_REVIEW
    report["release_gate"] = "human_continue_or_stop_required"
    report["requires_human_decision"] = True
    report["human_prompt"] = (
        "The previous output did not meet the citation credibility threshold. "
        "Rerun with the reviewer feedback or remove unsupported claims before release."
    )
    report["summary"] = (
        f"Reviewer Agent still requested revision after {max_review_cycles} research cycle(s). "
        + str(report.get("summary", "") or "")
    ).strip()
    limits = dict(report.get("limits", {}) or {})
    limits["max_review_cycles"] = max_review_cycles
    report["limits"] = limits
    return report


def _generator_feedback_message(
    *,
    query: str,
    grounded: GroundedAnswer,
    answer_audit: dict[str, Any],
    critic_report: dict[str, Any],
    cycle_index: int,
    max_review_cycles: int,
) -> str:
    payload = {
        "schema_version": "reviewer_feedback.v1",
        "control_flow": {
            "cycle_completed": cycle_index,
            "max_review_cycles": max_review_cycles,
            "next_cycle": cycle_index + 1,
            "instruction": "Run search_documents again if better evidence is needed, then return a concise retrieval status.",
        },
        "original_query": query,
        "previous_answer": _truncate(grounded.answer, 1200),
        "reviewer": {
            "status": critic_report.get("status", ""),
            "credibility_score": critic_report.get("credibility_score"),
            "threshold": critic_report.get("threshold"),
            "summary": critic_report.get("summary", ""),
            "overall_reason": critic_report.get("overall_reason", ""),
            "generator_feedback": critic_report.get("generator_feedback", ""),
            "suggested_search_queries": list(critic_report.get("suggested_search_queries", []) or []),
            "weak_citations": _weak_citation_feedback(critic_report),
        },
        "retrieval_context": _compact_retrieval_context(answer_audit),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    compacted = len(text) > MAX_REVIEW_FEEDBACK_CHARS
    if compacted:
        payload["previous_answer"] = _truncate(grounded.answer, 600)
        payload["reviewer"]["weak_citations"] = payload["reviewer"]["weak_citations"][:3]
        payload["retrieval_context"] = _compact_retrieval_context(answer_audit, source_limit=4, applied=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) > MAX_REVIEW_FEEDBACK_CHARS:
        text = text[: MAX_REVIEW_FEEDBACK_CHARS - 20] + "\n...truncated"
        compacted = True
    header = (
        "Reviewer Agent feedback for the Research Agent.\n"
        "Do not answer the user directly in this step. Use the feedback to improve retrieval planning.\n"
        "Do not use excluded source text. Preserve the original user question.\n"
    )
    remaining_chars = max(400, MAX_REVIEW_FEEDBACK_CHARS - len(header))
    if len(text) > remaining_chars:
        text = text[: max(0, remaining_chars - 20)] + "\n...truncated"
    return header + text


def _compact_retrieval_context(
    answer_audit: dict[str, Any],
    *,
    source_limit: int | None = None,
    applied: bool = True,
) -> dict[str, Any]:
    retrieval = answer_audit.get("retrieval", {}) if isinstance(answer_audit.get("retrieval"), dict) else {}
    sources = [source for source in retrieval.get("sources_sent_to_answer", []) or [] if isinstance(source, dict)]
    limit = source_limit if source_limit is not None else MAX_REVIEW_CONTEXT_SOURCES
    return {
        "context_compaction": {
            "applied": applied,
            "strategy": "metadata_only_source_lookup",
            "full_source_text_location": "ADK session state search_history_sources",
            "reason": "review feedback passes source identifiers and metadata instead of full tool results",
        },
        "search_queries": list(retrieval.get("search_queries", []) or [])[-6:],
        "source_count": len(sources),
        "sources": [_review_source_summary(source) for source in sources[:limit]],
        "excluded_source_summary": _restricted_exclusion_summary(retrieval.get("excluded_sources", []) or []),
    }


def _review_source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": source.get("chunk_id") or source.get("source_id") or "",
        "doc_id": source.get("doc_id", ""),
        "title": source.get("title", ""),
        "page": source.get("page", ""),
        "access_level": source.get("access_level", ""),
        "status": source.get("status", ""),
        "rerank_score": source.get("rerank_score"),
    }


def _weak_citation_feedback(critic_report: dict[str, Any]) -> list[dict[str, Any]]:
    weak: list[dict[str, Any]] = []
    for item in critic_report.get("citation_reviews", []) or []:
        if not isinstance(item, dict) or item.get("verdict") == "verified":
            continue
        weak.append(
            {
                "citation_index": item.get("citation_index"),
                "verdict": item.get("verdict", "unclear"),
                "answer_span": _truncate(str(item.get("answer_span", "") or ""), 220),
                "source_ids": list(item.get("source_ids", []) or []),
                "reason": _truncate(str(item.get("reason", "") or ""), 220),
            }
        )
        if len(weak) >= 6:
            break
    return weak


def _review_cycle_summary(
    *,
    cycle_index: int,
    max_review_cycles: int,
    critic_report: dict[str, Any],
    feedback_message: str,
    source_count: int,
    search_count: int,
) -> dict[str, Any]:
    return {
        "cycle": cycle_index,
        "max_review_cycles": max_review_cycles,
        "reviewer_status": critic_report.get("status", ""),
        "critic_status": critic_report.get("status", ""),
        "credibility_score": critic_report.get("credibility_score"),
        "release_gate": critic_report.get("release_gate", ""),
        "feedback_sent_to_research_agent": bool(feedback_message),
        "feedback_sent_to_generator": bool(feedback_message),
        "feedback_char_count": len(feedback_message),
        "feedback_preview": _truncate(str(critic_report.get("generator_feedback", "") or ""), 900),
        "suggested_search_queries": list(critic_report.get("suggested_search_queries", []) or [])[:4],
        "weak_citations": _weak_citation_feedback(critic_report),
        "source_count": source_count,
        "search_count": search_count,
    }


def _review_control_summary(
    *,
    max_review_cycles: int,
    review_cycles: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "review_control.v1",
        "pattern": "research_reviewer_bounded_loop",
        "research_agent": "defence_agent_research",
        "reviewer_agent": "defence_agent_reviewer",
        "generator_agent": "defence_agent_research",
        "critic_agent": "defence_agent_reviewer",
        "max_review_cycles": max_review_cycles,
        "completed_review_cycles": len(review_cycles),
        "cycles": list(review_cycles),
        "context_compaction": {
            "strategy": "metadata_only_source_lookup",
            "full_source_text_location": "ADK session state search_history_sources",
            "max_feedback_chars": MAX_REVIEW_FEEDBACK_CHARS,
        },
    }


def _truncate(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 12)].rstrip() + " ...truncated"


def _critic_skipped_report(*, route: RouteDecision, citation_count: int) -> dict[str, Any]:
    return {
        "schema_version": "reviewer_report.v1",
        "reviewer": "defence_agent_reviewer",
        "status": "not_run",
        "credibility_score": None,
        "threshold": 0.8,
        "verified_citation_count": None,
        "unverified_citation_count": None,
        "total_citation_count": citation_count,
        "citation_reviews": [],
        "release_gate": "not_applicable",
        "requires_human_decision": False,
        "human_prompt": "",
        "generator_feedback": "",
        "suggested_search_queries": [],
        "summary": f"Reviewer Agent skipped for {route.label}.",
        "overall_reason": "The selected demo route does not include Reviewer Agent scoring.",
        "raw_critic_response": {},
        "limits": {
            "truth_verification": "not_claimed",
            "human_approval": "not_claimed",
            "dynamic_multi_agent_fanout": "not_used",
        },
    }


async def _critic_report(
    *,
    query: str,
    answer_audit: dict[str, Any],
    grounded: GroundedAnswer,
    sources_sent_to_answer: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        return await review_answer(
            query=query,
            answer=grounded.answer,
            citations=grounded.citations,
            answer_audit=answer_audit,
            sources_sent_to_answer=sources_sent_to_answer,
        )
    except Exception as exc:
        return {
            "schema_version": "reviewer_report.v1",
            "reviewer": "defence_agent_reviewer",
            "status": NEEDS_HUMAN_REVIEW,
            "credibility_score": 0.0,
            "threshold": 0.8,
            "verified_citation_count": 0,
            "unverified_citation_count": len(grounded.citations),
            "total_citation_count": len(grounded.citations),
            "citation_reviews": [],
            "release_gate": "human_continue_or_stop_required",
            "requires_human_decision": True,
            "human_prompt": (
                "The previous output did not meet the citation credibility threshold. "
                "Rerun with the reviewer feedback or remove unsupported claims before release."
            ),
            "generator_feedback": "",
            "suggested_search_queries": [],
            "summary": f"Reviewer review failed: {type(exc).__name__}: {exc}",
            "overall_reason": f"{type(exc).__name__}: {exc}",
            "raw_critic_response": {},
            "limits": {
                "truth_verification": "not_claimed",
                "human_approval": "not_claimed",
                "dynamic_multi_agent_fanout": "not_used",
            },
        }


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
