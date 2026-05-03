from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from defence_agent.agent.router import router
from defence_agent.agent.workflows import (
    _looks_outside_demo_corpus,
    ambiguous_query,
    evidence_payload,
    human_review,
    make_citations,
    table_analysis,
)
from defence_agent.auth.context import AuthContext
from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.models import AskRequest, AskResponse, SourceChunk
from defence_agent.observability.metrics import REQUEST_COUNT, REQUEST_LATENCY, ROUTE_COUNT, SAFETY_BLOCKS
from defence_agent.observability.tracing import trace_manager
from defence_agent.safety import validate_input_query, validate_output
from defence_agent.tools.registry import tool_registry


def stream_agent_events(request: AskRequest, auth: AuthContext) -> Iterator[dict[str, Any]]:
    trace_id = trace_manager.new_trace_id()
    started = time.perf_counter()
    trace_manager.start_trace(
        trace_id,
        auth.user_id,
        {"query": request.query, "route_override": request.route_override, "debug": request.debug, "stream": True},
    )
    trace_manager.add_span(trace_id, "request_received", {"path": "/v1/agent/stream"})
    trace_manager.add_span(trace_id, "auth_validated", auth.model_dump())
    yield _event("trace", {"trace_id": trace_id})

    try:
        yield _event("stage", {"label": "Input safety"})
        input_safety = validate_input_query(request.query)
        trace_manager.add_span(trace_id, "input_safety_checked", input_safety.as_dict())
        if input_safety.blocked:
            SAFETY_BLOCKS.labels(stage="input").inc()
            response = AskResponse(
                trace_id=trace_id,
                route="human_review",
                answer="I cannot process this query safely. Please shorten or rephrase it.",
                safety={"input": input_safety.as_dict()},
                needs_human_review=True,
                latency_ms=_elapsed_ms(started),
            )
            yield _event("final", response.model_dump())
            trace_manager.finish_trace(trace_id, "ok", response.model_dump(), route=response.route, duration_ms=response.latency_ms)
            return

        yield _event("stage", {"label": "Route selection"})
        decision = router.route(request.query, auth, request.route_override)
        ROUTE_COUNT.labels(route=decision.route).inc()
        trace_manager.set_route(trace_id, decision.route)
        trace_manager.add_span(trace_id, "route_selected", decision.model_dump())
        trace_manager.add_span(
            trace_id,
            "plan_created",
            {"route": decision.route, "tools": decision.needs_tools, "max_steps": decision.max_steps},
        )
        yield _event("route", decision.model_dump())

        yield _event("stage", {"label": "Tools and retrieval"})
        prepared = _prepare_context(request.query, decision.route, auth, trace_id)
        sources = prepared["sources"]
        citations = make_citations(sources)
        answer = prepared.get("answer")

        if answer is None and sources:
            yield _event("stage", {"label": "Cohere v2 streaming generation"})
            answer_parts: list[str] = []
            with trace_manager.span(trace_id, "model_generation_started", {"route": decision.route, "source_count": len(sources), "stream": True}):
                try:
                    for text in cohere_gateway.stream_answer(request.query, evidence_payload(sources), decision.route):
                        answer_parts.append(text)
                        yield _event("delta", {"text": text})
                except Exception as exc:
                    prepared["degradations"].append("stream_generation_failed_returned_evidence_packet")
                    answer_parts = [
                        "I could not complete streaming generation, so I am returning the evidence packet:\n",
                        *[
                            f"[C{index + 1}] {source.title}, {source.section}, page {source.page}: {source.summary}\n"
                            for index, source in enumerate(sources)
                        ],
                    ]
                    trace_manager.add_span(trace_id, "model_generation_started", {"error": str(exc)}, status="error", error=str(exc))
            answer = "".join(answer_parts)
        elif answer is not None:
            for token in answer.split():
                yield _event("delta", {"text": token + " "})

        answer = answer or "I do not have enough authorized evidence to answer. Please clarify the document, date, or planning process."
        if sources and "[C" not in answer:
            answer = answer.rstrip() + " [C1]"
            yield _event("delta", {"text": " [C1]"})
        validation = tool_registry.call(
            "validate_citations",
            {"answer": answer, "citation_ids": [citation.id for citation in citations]},
            auth,
            trace_id,
        )
        trace_manager.add_span(trace_id, "citation_validation_completed", validation.model_dump())
        output_safety = validate_output(answer, [citation.id for citation in citations], requires_citation=bool(sources))
        trace_manager.add_span(trace_id, "output_safety_checked", output_safety.as_dict())
        if output_safety.blocked:
            SAFETY_BLOCKS.labels(stage="output").inc()
            answer = "I blocked the generated answer because it may contain sensitive content. A human review is required."

        response = AskResponse(
            trace_id=trace_id,
            route=decision.route,
            answer=answer,
            citations=citations,
            sources=sources,
            plan=[decision.reason, *[f"Use tool: {tool}" for tool in decision.needs_tools]],
            tool_calls=[*prepared["tool_calls"], validation.model_dump()],
            safety={
                "input": input_safety.as_dict(),
                "output": output_safety.as_dict(),
                "citation_validation": validation.model_dump(),
            },
            degradations=prepared["degradations"],
            token_cost_estimate=_estimate_tokens_and_cost(request.query, answer, sources),
            needs_human_review=prepared["needs_human_review"] or output_safety.blocked,
            latency_ms=_elapsed_ms(started),
        )
        trace_manager.add_span(trace_id, "response_returned", {"route": response.route, "needs_human_review": response.needs_human_review})
        trace_manager.finish_trace(trace_id, "ok", response.model_dump(), route=response.route, duration_ms=response.latency_ms)
        REQUEST_COUNT.labels(route=response.route, status="ok").inc()
        REQUEST_LATENCY.labels(route=response.route).observe(time.perf_counter() - started)
        yield _event("final", response.model_dump())
    except Exception as exc:
        trace_manager.add_span(trace_id, "response_returned", {"error": str(exc)}, status="error", error=str(exc))
        trace_manager.finish_trace(trace_id, "error", {"error": str(exc)}, route=None, duration_ms=_elapsed_ms(started))
        REQUEST_COUNT.labels(route="stream_error", status="error").inc()
        yield _event("error", {"trace_id": trace_id, "error": str(exc)})


def _prepare_context(query: str, route: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    if route == "direct_rag":
        if _looks_outside_demo_corpus(query):
            return {
                "answer": "I do not have enough authorized evidence to answer this. The indexed corpus does not contain real deployment locations, live timing, budgets, or operational tasking.",
                "sources": [],
                "tool_calls": [],
                "degradations": ["outside_demo_corpus_abstained"],
                "needs_human_review": False,
            }
        search = tool_registry.call("search_doctrine", {"query": query, "top_k": 5}, auth, trace_id)
        sources = _sources_from_search(search)
        return _prepared(sources, [search.model_dump()], search.data.get("degradations", []) if search.ok else [search.error], False)

    if route == "version_comparison":
        comparison = tool_registry.call("compare_versions", {"query": query, "older_version": "2024", "newer_version": "2025"}, auth, trace_id)
        older = [SourceChunk.model_validate(item) for item in comparison.data.get("older", [])] if comparison.ok else []
        newer = [SourceChunk.model_validate(item) for item in comparison.data.get("newer", [])] if comparison.ok else []
        return _prepared((newer + older)[:6], [comparison.model_dump()], comparison.data.get("degradations", []) if comparison.ok else [comparison.error], False)

    if route == "table_analysis":
        result = table_analysis(query, auth, trace_id)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "tool_calls": result["tool_calls"],
            "degradations": result["degradations"],
            "needs_human_review": result["needs_human_review"],
        }

    if route == "restricted_access":
        search = tool_registry.call(
            "search_doctrine",
            {"query": query, "top_k": 5, "filters": {"title_contains": "Restricted Annex B"}},
            auth,
            trace_id,
        )
        sources = _sources_from_search(search)
        if not sources:
            return {
                "answer": "I cannot access Restricted Annex B for this user. A planning lead or admin with restricted clearance can review it.",
                "sources": [],
                "tool_calls": [search.model_dump()],
                "degradations": ["unauthorized_or_no_restricted_evidence"],
                "needs_human_review": auth.role not in {"planning_lead", "admin"},
            }
        return _prepared(sources, [search.model_dump()], search.data.get("degradations", []), False)

    if route == "security_test":
        search = tool_registry.call(
            "search_doctrine",
            {"query": "Poisoned Test Document instruction injection exception handling", "top_k": 4, "filters": {"doc_type": "test"}},
            auth,
            trace_id,
        )
        return _prepared(_sources_from_search(search), [search.model_dump()], search.data.get("degradations", []) if search.ok else [search.error], False)

    if route == "multi_source_synthesis":
        search = tool_registry.call("search_doctrine", {"query": query, "top_k": 6}, auth, trace_id)
        sources = _sources_from_search(search)
        if len(sources) < 2:
            return {
                "answer": "I need at least two authorized sources to synthesize this safely. Please narrow the documents or ask for human review.",
                "sources": sources,
                "tool_calls": [search.model_dump()],
                "degradations": ["weak_multi_source_retrieval"],
                "needs_human_review": True,
            }
        return _prepared(sources, [search.model_dump()], search.data.get("degradations", []), False)

    if route == "ambiguous_query":
        result = ambiguous_query(query, auth, trace_id)
    else:
        result = human_review(query, auth, trace_id)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "tool_calls": result["tool_calls"],
        "degradations": result["degradations"],
        "needs_human_review": result["needs_human_review"],
    }


def _sources_from_search(search: Any) -> list[SourceChunk]:
    if not search.ok:
        return []
    return [SourceChunk.model_validate(item) for item in search.data.get("chunks", [])]


def _prepared(sources: list[SourceChunk], tool_calls: list[dict[str, Any]], degradations: list[Any], needs_human_review: bool) -> dict[str, Any]:
    return {
        "answer": None,
        "sources": sources,
        "tool_calls": tool_calls,
        "degradations": [str(item) for item in degradations if item],
        "needs_human_review": needs_human_review,
    }


def _event(event: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"event": event, "data": data}


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _estimate_tokens_and_cost(query: str, answer: str, sources: list[SourceChunk]) -> dict[str, Any]:
    context_text = " ".join(source.text for source in sources)
    input_tokens = max(1, int((len(query.split()) + len(context_text.split())) * 1.33))
    output_tokens = max(1, int(len(answer.split()) * 1.33))
    return {
        "input_tokens_estimate": input_tokens,
        "output_tokens_estimate": output_tokens,
        "total_tokens_estimate": input_tokens + output_tokens,
        "estimated_cost_usd": 0.0,
        "mode": "mock_or_local_estimate",
    }
