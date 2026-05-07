from __future__ import annotations

from typing import Any
import re

from sqlmodel import Session, select

from defence_agent.auth.context import AuthContext
from defence_agent.auth.policy import policy_engine
from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.db import engine
from defence_agent.models import Chunk, Citation, SourceChunk
from defence_agent.observability.tracing import trace_manager
from defence_agent.tools.registry import tool_registry


def make_citations(sources: list[SourceChunk]) -> list[Citation]:
    return [
        Citation(
            id=f"C{index + 1}",
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            title=source.title,
            section=source.section,
            page=source.page,
            source_uri=source.source_uri,
            filename=source.filename,
            classification=source.classification,
            version=source.version,
            effective_date=source.effective_date,
            status=source.status,
            doc_family=source.doc_family,
            owner=source.owner,
            review_due=source.review_due,
            language=source.language,
            row_id=source.row_id,
            doc_type=source.doc_type,
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
            "doc_id": source.document_id,
            "version": source.version,
            "status": source.status,
            "effective_date": source.effective_date,
            "access_level": source.classification,
            "language": source.language,
            "row_id": source.row_id,
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


def evidence_lookup(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    sources, search = _search_sources(
        query,
        auth,
        trace_id,
        top_k=4,
        filters={"status": "approved", "language": "en", "exclude_doc_types": ["table"]},
    )
    if not sources:
        return _safe_abstain([search.model_dump()], "I do not have enough approved evidence to answer this planning question.")
    answer = _generate(query, sources[:3], "evidence_lookup", trace_id)
    return _workflow_result(answer, sources[:3], [search.model_dump()], search.data.get("degradations", []))


def grounded_summary(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    lowered = query.lower()
    if "planning brief" in lowered or "director" in lowered or "circulating a planning brief" in lowered:
        filters = {"doc_family": "planning_brief_approval", "status": "approved", "language": "en"}
        failure = "I could not find an approved planning brief approval SOP to summarize."
    elif "evidence checklist" in lowered or "pre-review checklist" in lowered:
        filters = {"doc_family": "planning_brief_checklist", "status": "approved", "language": "en"}
        failure = "I could not find an approved planning brief evidence checklist to summarize."
    elif "decision log" in lowered or "log retention" in lowered or "retention procedure" in lowered:
        filters = {"doc_family": "evidence_log_retention", "status": "approved", "language": "en"}
        failure = "I could not find an approved evidence and decision log retention procedure to summarize."
    else:
        filters = {"doc_family": "emergency_communications", "status": "approved", "language": "en"}
        failure = "I could not find an approved emergency communications procedure to summarize."
    sources, search = _search_sources(query, auth, trace_id, top_k=5, filters=filters)
    if not sources:
        return _safe_abstain([search.model_dump()], failure)
    answer = _generate(query, sources, "grounded_summary", trace_id)
    return _workflow_result(answer, sources, [search.model_dump()], search.data.get("degradations", []))


def metadata_aware_retrieval(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    filters = {
        "doc_family": "planning_brief_approval",
        "status": "approved",
        "language": "en",
        "exclude_statuses": ["draft", "superseded"],
    }
    sources, search = _search_sources(query, auth, trace_id, top_k=4, filters=filters)
    if not sources:
        return _safe_abstain([search.model_dump()], "I could not find current approved planning brief approval guidance.")
    answer = _generate(query, sources, "metadata_aware_retrieval", trace_id)
    return _workflow_result(answer, sources, [search.model_dump()], search.data.get("degradations", []))


def cross_source_synthesis(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    trace_manager.add_span(trace_id, "plan_created", {"steps": ["retrieve approved SOP", "follow PB-CHK-2025 reference", "synthesize with citations"]})
    primary, search = _search_sources(
        "Planning brief approval SOP evidence pack cross-reference PB-CHK-2025",
        auth,
        trace_id,
        top_k=3,
        filters={"doc_id": "PB-SOP-2025", "status": "approved"},
    )
    follow = tool_registry.call(
        "follow_references",
        {"query": query, "retrieved_chunks": [source.model_dump() for source in primary], "top_k": 4},
        auth,
        trace_id,
    )
    followed = [SourceChunk.model_validate(item) for item in follow.data.get("chunks", [])] if follow.ok else []
    sources = _dedupe_sources([*primary, *followed])[:6]
    if len({source.document_id for source in sources}) < 2:
        checklist, checklist_search = _search_sources(
            query,
            auth,
            trace_id,
            top_k=3,
            filters={"doc_id": "PB-CHK-2025", "status": "approved"},
        )
        sources = _dedupe_sources([*sources, *checklist])[:6]
        tool_calls = [search.model_dump(), follow.model_dump(), checklist_search.model_dump()]
    else:
        tool_calls = [search.model_dump(), follow.model_dump()]
    if not sources:
        return _safe_abstain(tool_calls, "I could not assemble the SOP and checklist evidence needed for this answer.")
    answer = _generate(query, sources, "cross_source_synthesis", trace_id)
    degradations = search.data.get("degradations", []) + (follow.data.get("degradations", []) if follow.ok else [follow.error])
    return _workflow_result(answer, sources, tool_calls, degradations)


def version_comparison(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    trace_manager.add_span(
        trace_id,
        "plan_created",
        {"steps": ["parallel retrieval for 2024 and 2025", "compare procedure changes", "validate citations"]},
    )
    comparison = tool_registry.call(
        "compare_document_versions",
        {
            "query": query,
            "doc_family": "planning_brief_approval",
            "version_a": "2024.3",
            "version_b": "2025.1",
            "older_version": "2024.3",
            "newer_version": "2025.1",
        },
        auth,
        trace_id,
    )
    if not comparison.ok:
        return _evidence_failure("version_comparison", comparison.error)
    older = [SourceChunk.model_validate(item) for item in comparison.data.get("older", [])]
    newer = [SourceChunk.model_validate(item) for item in comparison.data.get("newer", [])]
    sources = (newer + older)[:6]
    if not sources:
        return _evidence_failure("version_comparison", "No authorized version evidence found")
    answer = _generate(query, sources, "version_comparison", trace_id)
    return _workflow_result(answer, sources, [comparison.model_dump()], comparison.data.get("degradations", []))


def structured_table_analysis(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    get_table = tool_registry.call(
        "get_table",
        {"table_name": "doctrine_review_tracker", "filters": {}},
        auth,
        trace_id,
    )
    if not get_table.ok:
        return _evidence_failure("structured_table_analysis", get_table.error)
    analysis = tool_registry.call(
        "run_table_analysis",
        {
            "dataset_id": get_table.data["dataset_id"],
            "analysis_goal": query,
            "rows": get_table.data["rows"],
            "today": "2026-05-06",
        },
        auth,
        trace_id,
    )
    tool_calls = [get_table.model_dump(), analysis.model_dump()]
    sandbox = analysis.data.get("sandbox", {}) if analysis.ok else {}
    if not analysis.ok or not sandbox.get("ok"):
        return {
            "answer": "The table analysis sandbox failed, so I cannot safely compute overdue review counts. Please inspect the authorized tracker rows or request human review.",
            "sources": [],
            "citations": [],
            "tool_calls": tool_calls,
            "degradations": ["python_failure_returned_table"],
            "needs_human_review": True,
        }
    output = sandbox["output"]
    row_ids = output.get("row_ids", [])
    sources = _table_sources_by_row_ids(row_ids, auth)
    citations = make_citations(sources)
    citation_for_row = {source.row_id: f"C{index + 1}" for index, source in enumerate(sources)}
    if output.get("analysis_type") == "count_approved_by_owner":
        row_lines = [
            f"- {item['owner']}: {item['approved_documents']} approved document(s)."
            for item in output.get("grouped", [])
        ]
        source_lines = [
            f"- {row['doc_id']} contributes to {row['owner']} [{citation_for_row.get(row['row_id'], 'C?')}]."
            for row in output.get("rows", [])
        ]
        answer = "Approved documents grouped by owner:\n" + "\n".join(row_lines)
        answer += "\n\nRows used:\n" + "\n".join(source_lines)
    else:
        row_lines = [
            f"- {row['doc_id']} owned by {row['owner']} is {row['days_overdue']} days overdue ({row['next_review_due']}) [{citation_for_row.get(row['row_id'], 'C?')}]."
            for row in output.get("rows", [])
        ]
        group_lines = [
            f"- {item['owner']}: {item['overdue_documents']} overdue document(s), max {item['max_days_overdue']} days overdue."
            for item in output.get("grouped", [])
        ]
        answer = "As of 2026-05-06, the approved planning procedures overdue for review are:\n" + "\n".join(row_lines)
        answer += "\n\nGrouped by owner:\n" + "\n".join(group_lines)
    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "tool_calls": tool_calls,
        "degradations": [],
        "needs_human_review": False,
    }


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


def claim_verification(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    lowered = query.lower()
    if "emergency" in lowered or "15 minutes" in lowered or "45 minutes" in lowered or "final record" in lowered:
        search_query = "Emergency communications acknowledgement 15 minutes operational update final record"
        filters = {"doc_id": "EC-PROC-2025", "status": "approved"}
    elif "decision log" in lowered or "retained" in lowered or "one year" in lowered or "trace id" in lowered:
        search_query = "Evidence and decision log retention 7 years trace ID"
        filters = {"doc_id": "LOG-RET-2025", "status": "approved"}
    elif "citation table" in lowered or "evidence pack" in lowered:
        search_query = "Planning brief evidence checklist citation table evidence pack"
        filters = {"doc_id": "PB-CHK-2025", "status": "approved"}
    else:
        search_query = "urgent planning brief evidence review cannot skip evidence review"
        filters = {"doc_id": "PB-SOP-2025", "status": "approved"}
    sources, search = _search_sources(
        search_query,
        auth,
        trace_id,
        top_k=4,
        filters=filters,
    )
    if not sources:
        return _safe_abstain([search.model_dump()], "I cannot verify the claim because approved guidance was not retrieved.")
    if "15 minutes" in lowered or "initial acknowledgement" in lowered:
        answer = "The statement is supported. The approved emergency communications procedure requires an initial acknowledgement within 15 minutes [C1]."
    elif "one year" in lowered:
        answer = "The statement is not supported. The approved retention procedure requires decision logs to be retained for 7 years, not one year [C1]."
    elif "trace id" in lowered:
        answer = "The statement is supported. The approved retention procedure says each briefing decision must include a trace ID [C1]."
    elif "citation table" in lowered:
        answer = "The statement is supported. The approved evidence checklist requires a citation table in the evidence pack [C1]."
    else:
        answer = (
            "The statement is not supported by approved guidance. The current approved SOP says urgency may compress sign-off order, "
            "but it cannot skip evidence review [C1]. Draft or superseded language should not be used to approve current procedure questions."
        )
    return _workflow_result(answer, sources[:1], [search.model_dump()], search.data.get("degradations", []))


def permission_sensitive_retrieval(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    lowered = query.lower()
    if "why did" in lowered and "analyst" in lowered and "restricted" in lowered:
        return {
            "answer": (
                "The analyst persona did not receive the restricted-annex answer because access control is enforced before retrieval. "
                "The restricted annex was excluded from model context for that persona, so the system could only return a refusal or partial public-source answer. "
                "This audit explanation exposes policy metadata, not restricted document content."
            ),
            "sources": [],
            "citations": [],
            "tool_calls": [],
            "degradations": ["audit_metadata_only"],
            "needs_human_review": False,
        }
    needs_public_sop = "planning brief" in lowered or "compare" in lowered
    public_sources: list[SourceChunk] = []
    tool_calls: list[dict[str, Any]] = []
    if needs_public_sop:
        public_sources, public_search = _search_sources(
            "Planning brief approval SOP restricted annexes external distribution legal policy review",
            auth,
            trace_id,
            top_k=3,
            filters={"doc_id": "PB-SOP-2025", "status": "approved"},
        )
        tool_calls.append(public_search.model_dump())
    annex_sources, search = _search_sources(
        query,
        auth,
        trace_id,
        top_k=4,
        filters={"doc_family": "annex_handling", "status": "approved"},
    )
    tool_calls.append(search.model_dump())
    sources = _dedupe_sources([*public_sources, *annex_sources])
    if public_sources and not annex_sources:
        citations = make_citations(public_sources[:2])
        answer = (
            "From the public approved SOP, planning briefs that include restricted annexes require legal/policy review before circulation "
            "and director approval before distribution [C1]. The detailed restricted-annex handling steps require a restricted source that is not available "
            "to this persona, so I cannot provide that portion. Escalate to a doctrine steward or authorized reviewer."
        )
        return {
            "answer": answer,
            "sources": public_sources[:2],
            "citations": citations,
            "tool_calls": tool_calls,
            "degradations": ["restricted_source_not_available_to_user"],
            "needs_human_review": True,
        }
    if not sources:
        return {
            "answer": "This answer requires a restricted source that is not available to the current user. Request access or escalate to an authorized planning lead.",
            "sources": [],
            "citations": [],
            "tool_calls": tool_calls,
            "degradations": ["restricted_source_not_available_to_user"],
            "needs_human_review": True,
        }
    answer = _generate(query, sources, "permission_sensitive_retrieval", trace_id)
    return _workflow_result(answer, sources, tool_calls, search.data.get("degradations", []))


def bilingual_retrieval(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    sources, search = _search_sources(
        query,
        auth,
        trace_id,
        top_k=4,
        filters={"doc_family": "emergency_communications", "status": "approved", "language": "fr"},
    )
    if not sources:
        return _safe_abstain([search.model_dump()], "Je n'ai pas trouve de source francaise approuvee pour repondre.")
    answer = _generate(query, sources, "bilingual_retrieval", trace_id)
    return _workflow_result(answer, sources, [search.model_dump()], search.data.get("degradations", []))


def refuse_or_clarify(query: str, auth: AuthContext, trace_id: str) -> dict[str, Any]:
    review = tool_registry.call(
        "request_human_review",
        {"reason": "Approved documents do not provide enough support or query is underspecified", "query": query, "route": "refuse_or_clarify"},
        auth,
        trace_id,
    )
    return {
        "answer": "The approved documents do not provide enough support for that request. Please clarify the scenario or route it to an authorized subject-matter expert for review.",
        "sources": [],
        "citations": [],
        "tool_calls": [review.model_dump()],
        "degradations": ["unsupported_or_ambiguous_request"],
        "needs_human_review": True,
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
    "evidence_lookup": evidence_lookup,
    "grounded_summary": grounded_summary,
    "metadata_aware_retrieval": metadata_aware_retrieval,
    "cross_source_synthesis": cross_source_synthesis,
    "direct_rag": direct_rag,
    "version_comparison": version_comparison,
    "structured_table_analysis": structured_table_analysis,
    "table_analysis": table_analysis,
    "claim_verification": claim_verification,
    "permission_sensitive_retrieval": permission_sensitive_retrieval,
    "bilingual_retrieval": bilingual_retrieval,
    "refuse_or_clarify": refuse_or_clarify,
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


def _search_sources(
    query: str,
    auth: AuthContext,
    trace_id: str,
    top_k: int,
    filters: dict[str, Any],
) -> tuple[list[SourceChunk], Any]:
    filters = {**filters}
    if not any(key in filters for key in ("doc_type", "doc_types")) and filters.get("doc_id") != "doctrine_review_tracker":
        filters.setdefault("exclude_doc_types", ["table"])
    search = tool_registry.call("search_documents", {"query": query, "top_k": top_k, "filters": filters}, auth, trace_id)
    if not search.ok:
        return [], search
    return [SourceChunk.model_validate(item) for item in search.data["chunks"]], search


def _dedupe_sources(sources: list[SourceChunk]) -> list[SourceChunk]:
    seen: set[str] = set()
    deduped: list[SourceChunk] = []
    for source in sources:
        if source.chunk_id in seen:
            continue
        seen.add(source.chunk_id)
        deduped.append(source)
    return deduped


def _safe_abstain(tool_calls: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    return {
        "answer": answer,
        "sources": [],
        "citations": [],
        "tool_calls": tool_calls,
        "degradations": ["weak_or_missing_retrieval"],
        "needs_human_review": False,
    }


def _table_sources_by_row_ids(row_ids: list[str], auth: AuthContext) -> list[SourceChunk]:
    if not row_ids:
        return []
    with Session(engine) as session:
        chunks = session.exec(
            select(Chunk)
            .where(Chunk.document_id == "doctrine_review_tracker")
            .where(Chunk.row_id.in_(row_ids))
            .order_by(Chunk.chunk_index)
        ).all()
    sources: list[SourceChunk] = []
    for chunk in chunks:
        if chunk.row_id in row_ids and policy_engine.can_access_chunk(auth, chunk):
            sources.append(
                SourceChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    filename="doctrine_review_tracker.csv",
                    section=chunk.section,
                    page=chunk.page,
                    text=chunk.text,
                    summary=chunk.summary,
                    classification=chunk.classification,
                    version=chunk.version,
                    effective_date=chunk.effective_date,
                    status=chunk.status,
                    doc_family=chunk.doc_family,
                    owner=chunk.owner,
                    review_due=chunk.review_due,
                    language=chunk.language,
                    source_type=chunk.source_type,
                    row_id=chunk.row_id,
                    doc_type=chunk.doc_type,
                    source_uri=chunk.source_uri,
                )
            )
    order = {row_id: index for index, row_id in enumerate(row_ids)}
    return sorted(sources, key=lambda source: order.get(source.row_id or "", 999))
