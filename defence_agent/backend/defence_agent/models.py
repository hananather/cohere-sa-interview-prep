from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Document(SQLModel, table=True):
    id: str = SQLField(primary_key=True)
    title: str
    filename: str
    doc_type: str
    classification: str
    allowed_roles_json: str
    version: str
    effective_date: str
    doc_family: str = SQLField(default="")
    status: str = SQLField(default="approved", index=True)
    owner: str = SQLField(default="")
    review_due: str = SQLField(default="")
    language: str = SQLField(default="en", index=True)
    source_type: str = SQLField(default="")
    checksum: str = SQLField(default="")
    parser_status: str = "ok"
    parser_confidence: float = 0.95
    created_at: datetime = SQLField(default_factory=utc_now)


class Chunk(SQLModel, table=True):
    id: str = SQLField(primary_key=True)
    document_id: str = SQLField(index=True, foreign_key="document.id")
    chunk_index: int
    title: str
    section: str
    page: int
    text: str
    table_markdown: str | None = None
    summary: str
    keywords_json: str
    classification: str
    allowed_roles_json: str
    tenant_id: str = SQLField(default="deftech", index=True)
    version: str
    effective_date: str
    doc_family: str = SQLField(default="", index=True)
    status: str = SQLField(default="approved", index=True)
    owner: str = SQLField(default="")
    review_due: str = SQLField(default="")
    language: str = SQLField(default="en", index=True)
    source_type: str = SQLField(default="")
    row_id: str | None = SQLField(default=None, index=True)
    doc_type: str
    source_uri: str
    embedding_json: str | None = None
    parser_status: str = "ok"
    parser_confidence: float = 0.95
    created_at: datetime = SQLField(default_factory=utc_now)


class TraceRecord(SQLModel, table=True):
    trace_id: str = SQLField(primary_key=True)
    user_id: str
    route: str | None = SQLField(default=None, index=True)
    status: str = "started"
    request_json: str
    response_json: str | None = None
    created_at: datetime = SQLField(default_factory=utc_now)
    duration_ms: float | None = None


class TraceSpan(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    trace_id: str = SQLField(index=True)
    name: str = SQLField(index=True)
    started_at: datetime = SQLField(default_factory=utc_now)
    ended_at: datetime | None = None
    duration_ms: float | None = None
    status: str = "ok"
    attributes_json: str = "{}"
    error: str | None = None


class Feedback(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    trace_id: str = SQLField(index=True)
    user_id: str
    helpful: bool
    comment: str | None = None
    created_at: datetime = SQLField(default_factory=utc_now)


class EvalRun(SQLModel, table=True):
    id: str = SQLField(primary_key=True)
    started_at: datetime = SQLField(default_factory=utc_now)
    completed_at: datetime | None = None
    metrics_json: str = "{}"
    status: str = "started"


class EvalCaseResult(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    run_id: str = SQLField(index=True)
    case_id: str = SQLField(index=True)
    slice: str
    passed: bool
    metrics_json: str = "{}"
    trace_id: str | None = None
    error: str | None = None


class SandboxRun(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    trace_id: str = SQLField(index=True)
    user_id: str
    code: str
    input_hash: str
    output_json: str | None = None
    error: str | None = None
    duration_ms: float
    created_at: datetime = SQLField(default_factory=utc_now)


class Citation(BaseModel):
    id: str
    chunk_id: str
    document_id: str | None = None
    title: str
    section: str
    page: int
    source_uri: str
    filename: str | None = None
    classification: str | None = None
    version: str | None = None
    effective_date: str | None = None
    status: str | None = None
    doc_family: str | None = None
    owner: str | None = None
    review_due: str | None = None
    language: str | None = None
    row_id: str | None = None
    doc_type: str | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str | None = None
    title: str
    filename: str | None = None
    section: str
    page: int
    text: str
    summary: str
    classification: str
    version: str
    effective_date: str
    status: str = "approved"
    doc_family: str = ""
    owner: str = ""
    review_due: str = ""
    language: str = "en"
    source_type: str = ""
    row_id: str | None = None
    doc_type: str
    source_uri: str
    lexical_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None
    table_markdown: str | None = None


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    route_override: str | None = None
    debug: bool = False


class AskResponse(BaseModel):
    trace_id: str
    route: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourceChunk] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    safety: dict[str, Any] = Field(default_factory=dict)
    degradations: list[str] = Field(default_factory=list)
    token_cost_estimate: dict[str, Any] = Field(default_factory=dict)
    needs_human_review: bool = False
    latency_ms: float | None = None
