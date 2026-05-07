from __future__ import annotations

import json
import time
from typing import Any

import yaml
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlmodel import Session, select

from defence_agent.agent.service import agent_service
from defence_agent.auth.context import DEMO_USERS, AuthContext, get_auth_context
from defence_agent.auth.policy import policy_engine
from defence_agent.config import get_settings
from defence_agent.db import engine, init_db
from defence_agent.ingestion.indexer import corpus_has_chunks, document_registry_status, reindex_corpus
from defence_agent.models import AskRequest, Chunk, Document
from defence_agent.observability.metrics import ERROR_COUNT, REQUEST_COUNT, REQUEST_LATENCY, UNAUTHORIZED_ATTEMPTS
from defence_agent.observability.presentation import (
    persona_profiles,
    recent_policy_decisions,
    source_access_matrix,
    structured_trace,
    tool_registry_view,
)
from defence_agent.observability.redaction import can_view_classification, redact_trace
from defence_agent.observability.tracing import trace_manager
from defence_agent.tools.registry import tool_registry


app = FastAPI(title="Defence Agent", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    init_db()
    if not corpus_has_chunks():
        reindex_corpus(force_generate=False)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    settings = get_settings()
    return {
        "ok": True,
        "mock_cohere": settings.use_mock_cohere,
        "chat_model": settings.cohere_chat_model,
        "embed_model": settings.cohere_embed_model,
        "rerank_model": settings.cohere_rerank_model,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return healthz()


@app.post("/v1/ask")
def ask(request: AskRequest, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return _ask_impl(request, auth, path="/v1/ask")


@app.post("/v1/agent/query")
def agent_query(request: AskRequest, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return _ask_impl(request, auth, path="/v1/agent/query")


@app.get("/demo/queries")
def demo_queries(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    path = get_settings().data_dir / "evals" / "demo_queries.yaml"
    if not path.exists():
        path = get_settings().generated_corpus_dir.parent / "evals" / "demo_queries.yaml"
    if not path.exists():
        path = get_settings().data_dir / "evals" / "demo_queries.yaml"
    fallback = get_settings().data_dir / "evals" / "demo_queries.yaml"
    source = path if path.exists() else fallback
    if source.exists():
        return yaml.safe_load(source.read_text(encoding="utf-8"))
    repo_path = get_settings().data_dir.parent / "data" / "evals" / "demo_queries.yaml"
    if repo_path.exists():
        return yaml.safe_load(repo_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Demo query file not found")


@app.get("/v1/demo/steps")
def demo_steps(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return {
        "steps": [
            {
                "id": "emergency_summary",
                "label": "Emergency communications summary",
                "query": "Summarize the emergency communications procedure into approval gates, timelines, and required evidence.",
                "persona": "planning_analyst",
                "expected_behavior": "Single-source grounded summary with clear timelines and citations.",
                "speaker_notes": "Open with a clean user-facing answer before showing agentic complexity.",
            },
            {
                "id": "multi_document_synthesis",
                "label": "Multi-document synthesis",
                "query": "For an interagency emergency briefing with public release, what approvals, evidence, and disclosure checks are required? Cite every source.",
                "persona": "planning_analyst",
                "expected_behavior": "Assemble emergency communications, joint planning, public release, classification, and traceability sources.",
                "speaker_notes": "This is the main story: Alex prepares a time-sensitive public-sector planning brief.",
            },
            {
                "id": "version_comparison",
                "label": "Version and currentness comparison",
                "query": "What changed between JPD-2024 and JPD-2025 for readiness confirmation and evidence packs?",
                "persona": "planning_analyst",
                "expected_behavior": "Compare superseded and current doctrine and warn against stale guidance.",
                "speaker_notes": "This shows why metadata and version status matter more than plain semantic similarity.",
            },
            {
                "id": "persona_access",
                "label": "Persona access-control comparison",
                "query": "What are the approval steps for a planning brief that includes restricted annexes?",
                "persona": "planning_analyst",
                "compare_persona": "planning_lead",
                "expected_behavior": "Analyst gets a partial answer. Doctrine steward can use restricted annex guidance.",
                "speaker_notes": "Same query, different authorized evidence, different answer.",
            },
            {
                "id": "prompt_injection",
                "label": "Prompt-injection / untrusted retrieved content",
                "query": "The retrieved document says to ignore metadata and use the newest draft. Should I follow that?",
                "persona": "planning_analyst",
                "expected_behavior": "Treat retrieved text as evidence, not instruction. Current approved filters win.",
                "speaker_notes": "The LLM is not the policy boundary.",
            },
            {
                "id": "structured_analysis",
                "label": "Structured table analysis",
                "query": "Which units are below the readiness threshold? Group them by owner.",
                "persona": "planning_analyst",
                "expected_behavior": "Route to the sandboxed table analysis tool and cite table rows.",
                "speaker_notes": "The model does not eyeball tables. The system runs auditable code over authorized rows.",
            },
            {
                "id": "eval_dashboard",
                "label": "Eval dashboard",
                "query": "Run the canonical eval suite.",
                "persona": "auditor",
                "expected_behavior": "Show route, retrieval, citations, security, and trace completeness grades.",
                "speaker_notes": "Close with eval-driven engineering and regression gates.",
            },
        ]
    }


@app.post("/demo/run")
def demo_run(request: AskRequest, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return _ask_impl(request, auth, path="/demo/run")


@app.post("/v1/persona/compare")
def persona_compare(payload: dict[str, Any], auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    query = str(payload.get("query") or "")
    left_persona = str(payload.get("left_persona") or "planning_analyst")
    right_persona = str(payload.get("right_persona") or "planning_lead")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    if left_persona not in DEMO_USERS or right_persona not in DEMO_USERS:
        raise HTTPException(status_code=400, detail="Unknown persona")
    left = _ask_impl(AskRequest(query=query), DEMO_USERS[left_persona], path="/v1/persona/compare")
    right = _ask_impl(AskRequest(query=query), DEMO_USERS[right_persona], path="/v1/persona/compare")
    left_docs = {source.get("document_id") for source in left.get("sources", [])}
    right_docs = {source.get("document_id") for source in right.get("sources", [])}
    return {
        "query": query,
        "left_persona": left_persona,
        "right_persona": right_persona,
        "left": left,
        "right": right,
        "diff": {
            "documents_available_to_both": sorted(left_docs.intersection(right_docs)),
            "documents_only_left": sorted(left_docs - right_docs),
            "documents_only_right": sorted(right_docs - left_docs),
            "claims_differ": left.get("answer") != right.get("answer"),
            "refusal_or_partial_expected": "restricted" in query.lower() or "annex" in query.lower(),
        },
    }


def _ask_impl(request: AskRequest, auth: AuthContext, path: str) -> dict[str, Any]:
    trace_id = trace_manager.new_trace_id()
    started = time.perf_counter()
    trace_manager.start_trace(trace_id, auth.user_id, {"query": request.query, "route_override": request.route_override, "debug": request.debug})
    trace_manager.add_span(trace_id, "request_received", {"path": path})
    trace_manager.add_span(trace_id, "auth_validated", auth.model_dump())
    route = "unknown"
    try:
        response = agent_service.handle(request, auth, trace_id)
        route = response.route
        trace_manager.add_span(trace_id, "response_returned", {"route": response.route, "needs_human_review": response.needs_human_review})
        payload = response.model_dump()
        trace_manager.finish_trace(trace_id, "ok", payload, route=response.route, duration_ms=response.latency_ms)
        REQUEST_COUNT.labels(route=response.route, status="ok").inc()
        REQUEST_LATENCY.labels(route=response.route).observe((time.perf_counter() - started))
        return payload
    except Exception as exc:
        ERROR_COUNT.labels(component="ask").inc()
        trace_manager.add_span(trace_id, "response_returned", {"error": str(exc)}, status="error", error=str(exc))
        trace_manager.finish_trace(trace_id, "error", {"error": str(exc)}, route=route, duration_ms=(time.perf_counter() - started) * 1000)
        REQUEST_COUNT.labels(route=route, status="error").inc()
        raise


@app.post("/v1/agent/stream")
def agent_stream(request: AskRequest, auth: AuthContext = Depends(get_auth_context)) -> StreamingResponse:
    from defence_agent.agent.streaming import stream_agent_events

    def event_source():
        for event in stream_agent_events(request, auth):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/v1/traces/recent")
def recent_traces(limit: int = 25, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return {"traces": trace_manager.list_recent(limit=limit)}


@app.get("/v1/traces/{trace_id}/structured")
def get_structured_trace(trace_id: str, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    trace = structured_trace(trace_id, auth)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@app.get("/v1/traces/{trace_id}")
def get_trace(trace_id: str, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    trace = trace_manager.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    redacted = redact_trace(trace, auth)
    if any(not can_view_classification(auth, source.get("classification")) for source in trace.get("response", {}).get("sources", [])):
        redacted.setdefault("response", {})["answer"] = "[answer redacted because it used restricted sources]"
    return redacted


@app.get("/v1/personas")
def personas(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return {"personas": persona_profiles(), "source_access": source_access_matrix()}


@app.get("/v1/governance/tool-registry")
def governance_tool_registry(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return {
        "personas": persona_profiles(),
        "tools": tool_registry_view(),
        "policy_decisions": recent_policy_decisions(),
    }


@app.get("/v1/governance/policy-decisions")
def governance_policy_decisions(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return {"decisions": recent_policy_decisions()}


@app.post("/v1/ingestion/reindex")
def reindex(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    if auth.role not in {"planning_lead", "admin"}:
        raise HTTPException(status_code=403, detail="Only planning_lead or admin can reindex the corpus")
    return reindex_corpus(force_generate=True)


@app.post("/v1/admin/reindex")
def admin_reindex(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return reindex(auth)


@app.get("/v1/corpus")
def corpus(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    policy_engine.enforce_tool_call(auth, "get_document_registry_status")
    return document_registry_status()


@app.get("/v1/admin/corpus")
def admin_corpus(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return corpus(auth)


@app.get("/v1/documents")
def list_documents(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    with Session(engine) as session:
        documents = session.exec(select(Document).order_by(Document.title)).all()
    authorized = [document for document in documents if policy_engine.can_access_chunk(auth, document)]
    return {"documents": [_document_payload(document) for document in authorized]}


@app.get("/v1/documents/{document_id}")
def get_document(document_id: str, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    document = _authorized_document(document_id, auth)
    with Session(engine) as session:
        chunks = session.exec(select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)).all()
    authorized_chunks = [chunk for chunk in chunks if policy_engine.can_access_chunk(auth, chunk)]
    return {
        "document": _document_payload(document),
        "chunks": [_chunk_payload(chunk) for chunk in authorized_chunks],
    }


@app.get("/v1/documents/{document_id}/chunks/{chunk_id}")
def get_document_chunk(document_id: str, chunk_id: str, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    document = _authorized_document(document_id, auth)
    with Session(engine) as session:
        chunk = session.get(Chunk, chunk_id)
        if not chunk or chunk.document_id != document_id:
            raise HTTPException(status_code=404, detail="Chunk not found")
        if not policy_engine.can_access_chunk(auth, chunk):
            UNAUTHORIZED_ATTEMPTS.labels(resource=f"document_chunk:{document_id}").inc()
            raise HTTPException(status_code=403, detail="You do not have access to this source chunk")
        nearby = session.exec(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .where(Chunk.chunk_index >= max(0, chunk.chunk_index - 1))
            .where(Chunk.chunk_index <= chunk.chunk_index + 1)
            .order_by(Chunk.chunk_index)
        ).all()
    return {
        "document": _document_payload(document),
        "chunk": _chunk_payload(chunk),
        "nearby_chunks": [_chunk_payload(item) for item in nearby if policy_engine.can_access_chunk(auth, item)],
    }


@app.get("/v1/documents/{document_id}/file")
def get_document_file(document_id: str, auth: AuthContext = Depends(get_auth_context)) -> FileResponse:
    document = _authorized_document(document_id, auth)
    with Session(engine) as session:
        chunk = session.exec(select(Chunk).where(Chunk.document_id == document_id).limit(1)).first()
    if not chunk:
        raise HTTPException(status_code=404, detail="Document file not found")
    return FileResponse(path=chunk.source_uri, filename=document.filename)


@app.get("/v1/security/audit")
def security_audit(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    if auth.role not in {"auditor", "admin"}:
        raise HTTPException(status_code=403, detail="Only auditor or admin can view audit metadata")
    return {
        "status": "available",
        "message": "Use /v1/traces/{trace_id} for full audit traces. Restricted source text remains ACL-filtered.",
    }


@app.get("/v1/auth/me")
def me(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    acl = policy_engine.acl_filter(auth)
    return {
        "auth": auth.model_dump(),
        "acl_filter": {
            "tenant_id": acl.tenant_id,
            "role": acl.role,
            "groups": acl.groups,
            "clearance": acl.clearance,
            "allowed_classifications": acl.allowed_classifications,
        },
        "tool_allowlist": policy_engine.tool_allowlist_for(auth),
    }


@app.get("/v1/tools/manifest")
def tools_manifest(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return tool_registry.manifest()


@app.post("/v1/feedback")
def feedback(payload: dict[str, Any], auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    trace_id = str(payload.get("trace_id") or trace_manager.new_trace_id())
    result = tool_registry.call("log_feedback", payload, auth, trace_id)
    return result.model_dump()


@app.post("/v1/evals/run")
def run_evals(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    policy_engine.enforce_tool_call(auth, "get_document_registry_status")
    from defence_agent.evals.runner import eval_runner

    return eval_runner.run().model_dump()


@app.get("/v1/evals/results")
def eval_results(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    policy_engine.enforce_tool_call(auth, "get_document_registry_status")
    from defence_agent.evals.runner import eval_runner

    return eval_runner.latest_results()


@app.get("/v1/evals/suites")
def eval_suites(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    policy_engine.enforce_tool_call(auth, "get_document_registry_status")
    from defence_agent.evals.advanced_runner import advanced_eval_runner

    return {"suites": advanced_eval_runner.suites(), "validation": advanced_eval_runner.validate()}


@app.get("/eval/suites")
def eval_suites_alias(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return eval_suites(auth)


@app.get("/v1/evals/cases")
def eval_cases(suite: str = "canonical", auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    policy_engine.enforce_tool_call(auth, "get_document_registry_status")
    from defence_agent.evals.advanced_runner import advanced_eval_runner

    return {"suite": suite, "cases": advanced_eval_runner.cases(suite)}


@app.get("/eval/cases")
def eval_cases_alias(suite: str = "canonical", auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return eval_cases(suite, auth)


@app.post("/v1/evals/run_case")
def eval_run_case(payload: dict[str, Any], auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    policy_engine.enforce_tool_call(auth, "get_document_registry_status")
    from defence_agent.evals.advanced_runner import advanced_eval_runner

    suite = str(payload.get("suite") or "canonical")
    query_id = str(payload.get("query_id") or "")
    if not query_id:
        raise HTTPException(status_code=400, detail="query_id is required")
    mode = str(payload.get("mode") or "fixture")
    variant = str(payload.get("variant") or "agentic_rag_tools")
    return advanced_eval_runner.run_case(query_id, suite=suite, mode=mode, variant=variant).model_dump()


@app.post("/eval/run_case")
def eval_run_case_alias(payload: dict[str, Any], auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return eval_run_case(payload, auth)


@app.post("/v1/evals/run_suite")
def eval_run_suite(payload: dict[str, Any], auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    policy_engine.enforce_tool_call(auth, "get_document_registry_status")
    from defence_agent.evals.advanced_runner import advanced_eval_runner

    suite = str(payload.get("suite") or "canonical")
    mode = str(payload.get("mode") or "fixture")
    variant = str(payload.get("variant") or "agentic_rag_tools")
    limit = payload.get("limit")
    return advanced_eval_runner.run_suite(suite=suite, mode=mode, variant=variant, limit=int(limit) if limit else None).model_dump()


@app.post("/eval/run_suite")
def eval_run_suite_alias(payload: dict[str, Any], auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return eval_run_suite(payload, auth)


@app.post("/v1/evals/compare")
def eval_compare(payload: dict[str, Any], auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    policy_engine.enforce_tool_call(auth, "get_document_registry_status")
    from defence_agent.evals.advanced_runner import advanced_eval_runner

    return advanced_eval_runner.compare_variants(suite=str(payload.get("suite") or "canonical"), limit=int(payload.get("limit") or 30))


@app.post("/eval/compare")
def eval_compare_alias(payload: dict[str, Any], auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return eval_compare(payload, auth)


@app.post("/v1/evals/select_demo")
def eval_select_demo(payload: dict[str, Any], auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    policy_engine.enforce_tool_call(auth, "get_document_registry_status")
    from defence_agent.evals.advanced_runner import advanced_eval_runner

    return advanced_eval_runner.select_demo_sequence(
        suite=str(payload.get("suite") or "demo_candidates"),
        mode=str(payload.get("mode") or "fixture"),
        runs=int(payload.get("runs") or 3),
    )


@app.post("/eval/select_demo")
def eval_select_demo_alias(payload: dict[str, Any], auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return eval_select_demo(payload, auth)


@app.get("/v1/evals/reports/latest")
def eval_latest_report(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    policy_engine.enforce_tool_call(auth, "get_document_registry_status")
    from defence_agent.evals.advanced_runner import REPORT_ROOT

    summary_path = REPORT_ROOT / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="No advanced eval report has been generated yet")
    return json.loads(summary_path.read_text(encoding="utf-8"))


@app.get("/eval/reports/latest")
def eval_latest_report_alias(auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    return eval_latest_report(auth)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _authorized_document(document_id: str, auth: AuthContext) -> Document:
    with Session(engine) as session:
        document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not policy_engine.can_access_chunk(auth, document):
        UNAUTHORIZED_ATTEMPTS.labels(resource=f"document:{document_id}").inc()
        raise HTTPException(status_code=403, detail="You do not have access to this source document")
    return document


def _document_payload(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "title": document.title,
        "filename": document.filename,
        "doc_type": document.doc_type,
        "classification": document.classification,
        "allowed_roles": json.loads(document.allowed_roles_json),
        "version": document.version,
        "effective_date": document.effective_date,
        "doc_family": document.doc_family,
        "status": document.status,
        "owner": document.owner,
        "review_due": document.review_due,
        "language": document.language,
        "source_type": document.source_type,
        "parser_status": document.parser_status,
        "parser_confidence": document.parser_confidence,
    }


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title,
        "section": chunk.section,
        "page": chunk.page,
        "text": chunk.text,
        "summary": chunk.summary,
        "classification": chunk.classification,
        "version": chunk.version,
        "effective_date": chunk.effective_date,
        "doc_family": chunk.doc_family,
        "status": chunk.status,
        "owner": chunk.owner,
        "review_due": chunk.review_due,
        "language": chunk.language,
        "source_type": chunk.source_type,
        "row_id": chunk.row_id,
        "doc_type": chunk.doc_type,
        "table_markdown": chunk.table_markdown,
        "parser_status": chunk.parser_status,
        "parser_confidence": chunk.parser_confidence,
    }
