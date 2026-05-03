from __future__ import annotations

from typing import Any
import re

from defence_agent.auth.context import AuthContext
from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.models import Citation, SourceChunk
from defence_agent.observability.tracing import trace_manager
from defence_agent.tools.registry import tool_registry


def make_citations(sources: list[SourceChunk]) -> list[Citation]:
    return [
        Citation(
            id=f"C{index + 1}",
            chunk_id=source.chunk_id,
            title=source.title,
            section=source.section,
            page=source.page,
            source_uri=source.source_uri,
        )
        for index, source in enumerate(sources)
    ]


def evidence_payload(sources: list[SourceChunk]) -> list[dict[str, Any]]:
    return [
        {
            "citation_id": f"C{index + 1}",
            "chunk_id": source.chunk_id,
            "title": source.title,
            "section": source.section,
            "page": source.page,
            "text": source.text,
            "summary": source.summary,
        }
        for index, source in enumerate(sources)
    ]


def direct_rag(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    if _looks_outside_demo_corpus(query):
        return {
            "answer": "I do not have enough authorized evidence to answer this. The indexed corpus does not contain real deployment locations, live timing, budgets, or operational tasking.",
            "sources": [],
            "citations": [],
            "tool_calls": [],
            "degradations": ["outside_demo_corpus_abstained"],
            "needs_human_review": False,
        }
    search = tool_registry.call("search_doctrine", {"query": query, "top_k": 5}, auth, trace_id)
    if not search.ok:
        return _evidence_failure("direct_rag", search.error)
    sources = [SourceChunk.model_validate(item) for item in search.data["chunks"]]
    if not sources:
        return {
            "answer": "I do not have enough authorized evidence to answer. Please clarify the document, date, or planning process.",
            "sources": [],
            "citations": [],
            "tool_calls": [search.model_dump()],
            "degradations": ["weak_retrieval_clarify"],
            "needs_human_review": False,
        }
    answer = _generate(query, sources, "direct_rag", trace_id)
    return _workflow_result(answer, sources, [search.model_dump()], search.data.get("degradations", []))


def version_comparison(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    trace_manager.add_span(
        trace_id,
        "plan_created",
        {"steps": ["parallel retrieval for 2024 and 2025", "compare procedure changes", "validate citations"]},
    )
    comparison = tool_registry.call("compare_versions", {"query": query, "older_version": "2024", "newer_version": "2025"}, auth, trace_id)
    if not comparison.ok:
        return _evidence_failure("version_comparison", comparison.error)
    older = [SourceChunk.model_validate(item) for item in comparison.data.get("older", [])]
    newer = [SourceChunk.model_validate(item) for item in comparison.data.get("newer", [])]
    sources = (newer + older)[:6]
    if not sources:
        return _evidence_failure("version_comparison", "No authorized version evidence found")
    answer = _generate(query, sources, "version_comparison", trace_id)
    return _workflow_result(answer, sources, [comparison.model_dump()], comparison.data.get("degradations", []))


def table_analysis(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    search = tool_registry.call(
        "search_doctrine",
        {
            "query": "readiness review table threshold unit readiness_pct open_actions",
            "top_k": 5,
            "filters": {"doc_type": "table"},
        },
        auth,
        trace_id,
    )
    if not search.ok:
        return _evidence_failure("table_analysis", search.error)
    sources = [SourceChunk.model_validate(item) for item in search.data["chunks"]]
    table_source = next((source for source in sources if source.table_markdown), None)
    if not table_source:
        return {
            "answer": "I found the readiness policy but not an authorized table block. Use the registry to reindex, or request human review.",
            "sources": sources,
            "citations": make_citations(sources),
            "tool_calls": [search.model_dump()],
            "degradations": ["table_block_missing"],
            "needs_human_review": True,
        }
    analysis = tool_registry.call(
        "analyze_table_with_python",
        {"table_markdown": table_source.table_markdown, "threshold": 80},
        auth,
        trace_id,
    )
    tool_calls = [search.model_dump(), analysis.model_dump()]
    citations = make_citations([table_source])
    if not analysis.ok or not analysis.data.get("sandbox", {}).get("ok"):
        return {
            "answer": f"Python analysis failed, so I am returning the authorized table for review [C1].",
            "sources": [table_source],
            "citations": citations,
            "tool_calls": tool_calls,
            "degradations": ["python_failure_returned_table"],
            "needs_human_review": True,
        }
    rows = analysis.data["sandbox"]["output"].get("below_threshold", [])
    if not rows:
        answer = "No units fall below the 80 percent readiness threshold in the authorized table [C1]."
    else:
        units = ", ".join(f"{row['unit']} ({row['readiness_pct']}%)" for row in rows)
        answer = f"The units below the 80 percent readiness threshold are {units}. They require mitigation owners before approval can continue [C1]."
    return {
        "answer": answer,
        "sources": [table_source],
        "citations": citations,
        "tool_calls": tool_calls,
        "degradations": search.data.get("degradations", []),
        "needs_human_review": False,
    }


def multi_source_synthesis(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    search = tool_registry.call("search_doctrine", {"query": query, "top_k": 6}, auth, trace_id)
    if not search.ok:
        return _evidence_failure("multi_source_synthesis", search.error)
    sources = [SourceChunk.model_validate(item) for item in search.data["chunks"]]
    if len(sources) < 2:
        return {
            "answer": "I need at least two authorized sources to synthesize this safely. Please narrow the documents or ask for human review.",
            "sources": sources,
            "citations": make_citations(sources),
            "tool_calls": [search.model_dump()],
            "degradations": ["weak_multi_source_retrieval"],
            "needs_human_review": True,
        }
    answer = _generate(query, sources, "multi_source_synthesis", trace_id)
    return _workflow_result(answer, sources, [search.model_dump()], search.data.get("degradations", []))


def ambiguous_query(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    review = tool_registry.call(
        "request_human_review",
        {"reason": "Query is ambiguous", "query": query, "route": "ambiguous_query"},
        auth,
        trace_id,
    )
    return {
        "answer": "I need more detail before answering. Please specify the document, year, unit, or planning workflow you mean.",
        "sources": [],
        "citations": [],
        "tool_calls": [review.model_dump()],
        "degradations": ["ambiguous_query_clarification"],
        "needs_human_review": False,
    }


def restricted_access(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    search = tool_registry.call(
        "search_doctrine",
        {"query": query, "top_k": 5, "filters": {"title_contains": "Restricted Annex B"}},
        auth,
        trace_id,
    )
    if not search.ok:
        return _evidence_failure("restricted_access", search.error)
    sources = [SourceChunk.model_validate(item) for item in search.data["chunks"]]
    if not sources:
        return {
            "answer": "I cannot access Restricted Annex B for this user. A planning lead or admin with restricted clearance can review it.",
            "sources": [],
            "citations": [],
            "tool_calls": [search.model_dump()],
            "degradations": ["unauthorized_or_no_restricted_evidence"],
            "needs_human_review": auth.role not in {"planning_lead", "admin"},
        }
    answer = _generate(query, sources, "restricted_access", trace_id)
    return _workflow_result(answer, sources, [search.model_dump()], search.data.get("degradations", []))


def security_test(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    search = tool_registry.call(
        "search_doctrine",
        {"query": "Poisoned Test Document instruction injection exception handling", "top_k": 4, "filters": {"doc_type": "test"}},
        auth,
        trace_id,
    )
    if not search.ok:
        return _evidence_failure("security_test", search.error)
    sources = [SourceChunk.model_validate(item) for item in search.data["chunks"]]
    if not sources:
        return _evidence_failure("security_test", "No authorized test document evidence found")
    answer = _generate(query, sources, "security_test", trace_id)
    return _workflow_result(answer, sources, [search.model_dump()], search.data.get("degradations", []))


def human_review(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    review = tool_registry.call(
        "request_human_review",
        {"reason": "Route requires human review", "query": query, "route": "human_review"},
        auth,
        trace_id,
    )
    return {
        "answer": "This request needs human review before I can answer safely.",
        "sources": [],
        "citations": [],
        "tool_calls": [review.model_dump()],
        "degradations": ["human_review_required"],
        "needs_human_review": True,
    }


WORKFLOWS = {
    "direct_rag": direct_rag,
    "version_comparison": version_comparison,
    "table_analysis": table_analysis,
    "multi_source_synthesis": multi_source_synthesis,
    "ambiguous_query": ambiguous_query,
    "restricted_access": restricted_access,
    "security_test": security_test,
    "human_review": human_review,
}


def _generate(query: str, sources: list[SourceChunk], route: str, trace_id: str) -> str:
    with trace_manager.span(trace_id, "model_generation_started", {"route": route, "source_count": len(sources)}):
        try:
            return cohere_gateway.generate_answer(query, evidence_payload(sources), route)
        except Exception:
            lines = [f"Evidence packet for `{route}`:"]
            for index, source in enumerate(sources, start=1):
                lines.append(f"[C{index}] {source.title}, {source.section}, page {source.page}: {source.summary}")
            return "\n".join(lines)


def _workflow_result(
    answer: str,
    sources: list[SourceChunk],
    tool_calls: list[dict[str, Any]],
    degradations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "sources": sources,
        "citations": make_citations(sources),
        "tool_calls": tool_calls,
        "degradations": degradations or [],
        "needs_human_review": False,
    }


def _evidence_failure(route: str, error: str | None) -> dict[str, Any]:
    return {
        "answer": f"I could not complete the {route} workflow. I am returning a safe fallback and asking for human review.",
        "sources": [],
        "citations": [],
        "tool_calls": [],
        "degradations": [error or "workflow_failed"],
        "needs_human_review": True,
    }


def _looks_outside_demo_corpus(query: str) -> bool:
    lowered = query.lower()
    return bool(
        re.search(
            r"\b(real deployment|deployment location|unit zulu|tomorrow|classified budget|actual budget|budget for 2026)\b",
            lowered,
        )
    )
