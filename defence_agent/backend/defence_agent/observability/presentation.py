from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from defence_agent.auth.context import DEMO_PERSONA_PROFILES, DEMO_USERS, AuthContext
from defence_agent.auth.policy import TOOL_ALLOWLIST, policy_engine
from defence_agent.db import engine
from defence_agent.models import Document
from defence_agent.observability.redaction import can_view_classification, debug_full_trace_enabled, redact_trace
from defence_agent.observability.tracing import trace_manager


TOOL_GOVERNANCE: dict[str, dict[str, Any]] = {
    "search_documents": {
        "description": "Search authorized document chunks with metadata filters and reranking.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["query", "filters", "candidate_count", "excluded_sources"],
    },
    "search_doctrine": {
        "description": "Legacy alias for authorized doctrine search.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["query", "filters", "candidate_count"],
    },
    "get_document_excerpt": {
        "description": "Open an authorized cited evidence passage.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["document_id", "chunk_id"],
    },
    "get_document_sections": {
        "description": "Retrieve full sections from an already authorized document.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["doc_id", "section_ids"],
    },
    "follow_references": {
        "description": "Resolve cross-referenced approved documents after initial retrieval.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["references", "max_depth"],
    },
    "compare_versions": {
        "description": "Compare authorized evidence across document versions.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["version_a", "version_b", "doc_family"],
    },
    "compare_document_versions": {
        "description": "Compare two document versions using authorized retrieval.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["version_a", "version_b", "doc_family"],
    },
    "verify_claim": {
        "description": "Validate whether an answer claim is supported by retrieved context.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["citation_ids", "validation_result"],
    },
    "validate_citations": {
        "description": "Validate that displayed citation IDs resolve to retrieved evidence.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["citation_ids", "valid"],
    },
    "validate_answer_citations": {
        "description": "Validate citations after grounded generation.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["citation_ids", "valid"],
    },
    "get_table": {
        "description": "Load authorized structured table rows.",
        "risk_level": "read_only",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["table_name", "filters", "row_count"],
    },
    "run_table_analysis": {
        "description": "Run controlled pandas analysis over already-authorized table rows.",
        "risk_level": "sandboxed_medium",
        "approval_required": False,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["sandbox_policy", "dataset_hash", "row_ids"],
    },
    "export_brief_draft": {
        "description": "Export a human-reviewed brief draft.",
        "risk_level": "reversible_write",
        "approval_required": True,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["draft_id", "approver"],
        "enabled_in_demo": False,
    },
    "admin_reindex": {
        "description": "Rebuild local corpus indexes.",
        "risk_level": "admin",
        "approval_required": True,
        "policy_callback": "PolicyGuardPlugin.before_tool",
        "trace_fields": ["corpus_version", "chunk_count"],
        "enabled_in_demo": False,
    },
}


def persona_profiles() -> list[dict[str, Any]]:
    profiles = []
    for persona_id, profile in DEMO_PERSONA_PROFILES.items():
        auth = DEMO_USERS[persona_id]
        profiles.append(
            {
                **profile,
                "role": auth.role,
                "groups": auth.groups,
                "clearance": auth.clearance,
                "allowed_tools": _friendly_tools(policy_engine.tool_allowlist_for(auth)),
            }
        )
    return profiles


def persona_profile(persona_id: str) -> dict[str, Any]:
    return next((profile for profile in persona_profiles() if profile["persona_id"] == persona_id), persona_profiles()[0])


def tool_registry_view() -> list[dict[str, Any]]:
    tools = sorted(set(TOOL_ALLOWLIST) | set(TOOL_GOVERNANCE))
    rows: list[dict[str, Any]] = []
    for tool_name in tools:
        governance = TOOL_GOVERNANCE.get(tool_name, {})
        allowed_personas = sorted(
            persona_id for persona_id, auth in DEMO_USERS.items() if auth.role in TOOL_ALLOWLIST.get(tool_name, set())
        )
        rows.append(
            {
                "tool_name": tool_name,
                "description": governance.get("description", "Internal tool."),
                "risk_level": governance.get("risk_level", "read_only"),
                "allowed_personas": allowed_personas,
                "input_schema": governance.get("input_schema", "pydantic_model"),
                "output_schema": governance.get("output_schema", "ToolEnvelope"),
                "approval_required": governance.get("approval_required", False),
                "policy_callback": governance.get("policy_callback", "PolicyGuardPlugin.before_tool"),
                "trace_fields": governance.get("trace_fields", []),
                "enabled_in_demo": governance.get("enabled_in_demo", tool_name in TOOL_ALLOWLIST),
            }
        )
    return rows


def structured_trace(trace_id: str, auth: AuthContext) -> dict[str, Any] | None:
    raw = trace_manager.get_trace(trace_id)
    if not raw:
        return None
    trace = redact_trace(raw, auth)
    if _response_contains_hidden_sources(raw.get("response", {}), auth):
        trace.setdefault("response", {})["answer"] = "[answer redacted because it used restricted sources]"
        for source in trace.get("response", {}).get("sources", []):
            if not can_view_classification(auth, source.get("classification")):
                source["text"] = "[restricted content redacted]"
                source["summary"] = "[restricted content redacted]"
    response = trace.get("response", {})
    request = trace.get("request", {})
    spans = [_span_payload(span) for span in trace.get("spans", [])]
    policy_spans = [span for span in spans if span["type"] == "policy"]
    tool_spans = [span for span in spans if span["type"] == "tool"]
    retrieval = _retrieval_payload(response, trace, auth)
    policy = _policy_payload(auth, response, retrieval, policy_spans)
    citation_validation = _citation_validation(response)
    duration = trace.get("duration_ms") or sum(float(span.get("duration_ms") or 0) for span in spans)
    return {
        "trace_id": trace.get("trace_id"),
        "run_id": trace.get("trace_id"),
        "timestamp": trace.get("created_at"),
        "app_version": "0.1.0",
        "corpus_version": "synthetic-v1",
        "debug_full_trace": debug_full_trace_enabled(),
        "model_config": _model_config(),
        "user": {
            "persona_id": auth.user_id,
            "display_name": persona_profile(auth.user_id).get("display_name", auth.user_id),
            "access_level": persona_profile(auth.user_id).get("access_level", auth.clearance),
            "roles": [auth.role, *auth.groups],
        },
        "query": {
            "raw": request.get("query") or request.get("eval_case") or "",
            "normalized": str(request.get("query") or "").strip().lower(),
            "language": _language_from_response(response),
            "task_type": response.get("route") or trace.get("route"),
            "complexity_level": _complexity_from_route(response.get("route") or trace.get("route")),
        },
        "policy": policy,
        "spans": spans,
        "retrieval": retrieval,
        "generation": {
            "answer": response.get("answer", ""),
            "citations": response.get("citations", []),
            "citation_validation": citation_validation,
        },
        "eval": {
            "case_id": request.get("eval_case"),
            "passed": None,
            "graders": {},
        },
        "feedback": None,
        "summary": trace_summary(trace, policy, spans, response, duration),
        "raw": trace,
    }


def trace_summary(trace: dict[str, Any], policy: dict[str, Any], spans: list[dict[str, Any]], response: dict[str, Any], duration: float | None) -> str:
    tools = sorted({span.get("attributes", {}).get("tool") for span in spans if span.get("attributes", {}).get("tool")})
    source_count = len(response.get("sources", []))
    excluded_count = len(policy.get("reasons", []))
    valid = "yes" if _citation_validation(response).get("passed") else "no"
    return "\n".join(
        [
            f"Trace {trace.get('trace_id')}",
            f"User: {trace.get('user_id')}",
            f"Route: {trace.get('route')}",
            f"Policy: {policy.get('decision')}",
            f"Tools: {', '.join(tools) if tools else 'none'}",
            f"Sources sent to model: {source_count}",
            f"Sources excluded: {excluded_count}",
            f"Citations valid: {valid}",
            f"Latency: {int(duration or 0)} ms",
            "Failure category: none",
        ]
    )


def recent_policy_decisions(limit: int = 30) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for trace in trace_manager.list_recent(limit=limit):
        full = trace_manager.get_trace(trace["trace_id"]) or {}
        for span in full.get("spans", []):
            attrs = span.get("attributes", {})
            if span.get("name") in {"before_tool_policy", "auth_validated"} or "decision" in attrs:
                decisions.append(
                    {
                        "trace_id": trace["trace_id"],
                        "user": trace.get("user_id"),
                        "tool_or_source": attrs.get("tool") or attrs.get("source") or "authz",
                        "decision": attrs.get("decision") or span.get("status", "ok"),
                        "reason": attrs.get("reason"),
                        "timestamp": span.get("start_time") or trace.get("created_at"),
                    }
                )
    return decisions[:limit]


def source_access_matrix() -> list[dict[str, Any]]:
    with Session(engine) as session:
        documents = session.exec(select(Document).order_by(Document.title)).all()
    rows: list[dict[str, Any]] = []
    for document in documents:
        row = {
            "doc_id": document.id,
            "title": document.title,
            "classification": document.classification,
            "status": document.status,
            "language": document.language,
        }
        for persona_id, auth in DEMO_USERS.items():
            row[persona_id] = "content" if policy_engine.can_access_chunk(auth, document) else (
                "metadata" if DEMO_PERSONA_PROFILES.get(persona_id, {}).get("can_view_audit_metadata") else "hidden"
            )
        rows.append(row)
    return rows


def _span_payload(span: dict[str, Any]) -> dict[str, Any]:
    name = span.get("name", "")
    span_type = "tool" if "tool" in name else "retrieval" if "retrieval" in name or "search" in name else "model" if "model" in name else "policy" if "policy" in name or "auth" in name else "eval" if "eval" in name else "agent"
    return {
        "span_id": span.get("span_id"),
        "parent_span_id": None,
        "name": name,
        "type": span_type,
        "start_time": span.get("start_time"),
        "end_time": span.get("end_time"),
        "duration_ms": span.get("duration_ms") or 0,
        "status": span.get("status"),
        "inputs": {},
        "outputs": {},
        "attributes": span.get("attributes", {}),
        "error": span.get("error"),
    }


def _retrieval_payload(response: dict[str, Any], trace: dict[str, Any], auth: AuthContext) -> dict[str, Any]:
    sources = response.get("sources", [])
    final_context = [_source_projection(source) for source in sources]
    excluded = _excluded_sources_for_auth(auth, sources)
    return {
        "filters_applied": _filters_from_trace(trace),
        "keyword_candidates": [],
        "embedding_candidates": [],
        "merged_candidates": final_context,
        "reranked_candidates": final_context,
        "final_context": final_context,
        "excluded_sources": excluded,
    }


def _source_projection(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": source.get("chunk_id"),
        "doc_id": source.get("document_id"),
        "title": source.get("title"),
        "section": source.get("section"),
        "status": source.get("status"),
        "access_level": source.get("classification"),
        "vector_score": source.get("vector_score"),
        "rerank_score": source.get("rerank_score"),
        "candidate_source": "hybrid",
    }


def _excluded_sources_for_auth(auth: AuthContext, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible_doc_ids = {source.get("document_id") for source in sources}
    excluded: list[dict[str, Any]] = []
    with Session(engine) as session:
        documents = session.exec(select(Document).order_by(Document.title)).all()
    for document in documents:
        if document.id in visible_doc_ids:
            continue
        if not can_view_classification(auth, document.classification):
            excluded.append({"doc_id": document.id, "title": document.title, "reason": "access_denied"})
        elif document.status == "draft":
            excluded.append({"doc_id": document.id, "title": document.title, "reason": "draft_excluded"})
        elif document.status == "superseded":
            excluded.append({"doc_id": document.id, "title": document.title, "reason": "superseded_excluded"})
    return excluded[:12]


def _policy_payload(auth: AuthContext, response: dict[str, Any], retrieval: dict[str, Any], policy_spans: list[dict[str, Any]]) -> dict[str, Any]:
    has_sources = bool(response.get("sources"))
    restricted_needed = any(item.get("reason") == "access_denied" for item in retrieval.get("excluded_sources", []))
    if not has_sources and restricted_needed:
        decision = "refuse"
    elif restricted_needed:
        decision = "partial"
    else:
        decision = "allow"
    reasons = [item for item in retrieval.get("excluded_sources", []) if item.get("reason") == "access_denied"]
    for span in policy_spans:
        attrs = span.get("attributes", {})
        if attrs.get("decision") == "block":
            reasons.append({"tool": attrs.get("tool"), "reason": attrs.get("reason")})
            decision = "refuse"
    return {
        "allowed_doc_access": persona_profile(auth.user_id).get("allowed_doc_access", []),
        "allowed_tools": _friendly_tools(policy_engine.tool_allowlist_for(auth)),
        "required_approvals": [],
        "decision": decision,
        "reasons": reasons,
    }


def _citation_validation(response: dict[str, Any]) -> dict[str, Any]:
    answer = response.get("answer", "")
    citations = response.get("citations", [])
    cited_ids = {citation.get("id") for citation in citations}
    mentioned = set()
    for token in answer.split("[C")[1:]:
        mentioned.add("C" + token.split("]", 1)[0])
    failures = [citation_id for citation_id in mentioned if citation_id not in cited_ids]
    return {"passed": not failures and (bool(cited_ids) or not answer), "failures": failures}


def _filters_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for span in trace.get("spans", []):
        attrs = span.get("attributes", {})
        payload = attrs.get("payload", {})
        if isinstance(payload, dict) and isinstance(payload.get("filters"), dict):
            filters.update(payload["filters"])
    return filters


def _model_config() -> dict[str, str]:
    from defence_agent.config import get_settings

    settings = get_settings()
    return {
        "chat_model": settings.cohere_chat_model,
        "embed_model": settings.cohere_embed_model,
        "rerank_model": settings.cohere_rerank_model,
    }


def _language_from_response(response: dict[str, Any]) -> str:
    sources = response.get("sources", [])
    if sources and sources[0].get("language"):
        return str(sources[0]["language"])
    return "en"


def _response_contains_hidden_sources(response: dict[str, Any], auth: AuthContext) -> bool:
    return any(not can_view_classification(auth, source.get("classification")) for source in response.get("sources", []))


def _complexity_from_route(route: str | None) -> str:
    return {
        "evidence_lookup": "L1",
        "grounded_summary": "L2",
        "metadata_aware_retrieval": "L2",
        "cross_source_synthesis": "L3",
        "version_comparison": "L4",
        "structured_table_analysis": "L4",
        "permission_sensitive_retrieval": "L5",
        "claim_verification": "L5",
        "bilingual_retrieval": "L3",
    }.get(route or "", "L2")


def _friendly_tools(tools: list[str]) -> list[str]:
    replacements = {
        "get_document_sections": "get_document_excerpt",
        "validate_citations": "verify_claim",
        "validate_answer_citations": "verify_claim",
    }
    return sorted({replacements.get(tool, tool) for tool in tools})
