from __future__ import annotations

import json
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlmodel import Session, select

from defence_agent.agent.service import agent_service
from defence_agent.auth.context import AuthContext, get_auth_context
from defence_agent.auth.policy import policy_engine
from defence_agent.config import get_settings
from defence_agent.db import engine, init_db
from defence_agent.ingestion.indexer import corpus_has_chunks, document_registry_status, reindex_corpus
from defence_agent.models import AskRequest, Chunk, Document
from defence_agent.observability.metrics import ERROR_COUNT, REQUEST_COUNT, REQUEST_LATENCY, UNAUTHORIZED_ATTEMPTS
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


@app.get("/v1/traces/{trace_id}")
def get_trace(trace_id: str, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    trace = trace_manager.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


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
        "doc_type": chunk.doc_type,
        "table_markdown": chunk.table_markdown,
        "parser_status": chunk.parser_status,
        "parser_confidence": chunk.parser_confidence,
    }
