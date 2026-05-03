from __future__ import annotations

import json
import re
from typing import Any, Callable

from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import Session

from defence_agent.auth.context import AuthContext
from defence_agent.auth.policy import policy_engine
from defence_agent.db import engine
from defence_agent.ingestion.indexer import document_registry_status
from defence_agent.models import Feedback
from defence_agent.observability.metrics import TOOL_COUNT, UNAUTHORIZED_ATTEMPTS
from defence_agent.observability.tracing import trace_manager
from defence_agent.retrieval.hybrid import hybrid_retriever
from defence_agent.safety import citation_ids_from_answer, sanitize_tool_output
from defence_agent.sandbox.python_sandbox import run_sandboxed_python


class ToolEnvelope(BaseModel):
    ok: bool
    tool: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class SearchDoctrineInput(BaseModel):
    query: str
    top_k: int = Field(default=6, ge=1, le=12)
    filters: dict[str, Any] = Field(default_factory=dict)


class CompareVersionsInput(BaseModel):
    query: str
    older_version: str = "2024"
    newer_version: str = "2025"
    top_k: int = Field(default=4, ge=1, le=8)


class AnalyzeTableInput(BaseModel):
    table_markdown: str
    threshold: float = 80
    code: str | None = None


class CitationValidationInput(BaseModel):
    answer: str
    citation_ids: list[str]


class HumanReviewInput(BaseModel):
    reason: str
    query: str
    route: str


class FeedbackInput(BaseModel):
    trace_id: str
    helpful: bool
    comment: str | None = None


class EmptyInput(BaseModel):
    pass


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[type[BaseModel], Callable[[BaseModel, AuthContext, str], dict[str, Any]], str]] = {
            "search_doctrine": (
                SearchDoctrineInput,
                self._search_doctrine,
                "Search authorized doctrine, manuals, tables, and bulletins.",
            ),
            "compare_versions": (
                CompareVersionsInput,
                self._compare_versions,
                "Compare authorized chunks across two document versions.",
            ),
            "analyze_table_with_python": (
                AnalyzeTableInput,
                self._analyze_table_with_python,
                "Run sandboxed Python against authorized table data.",
            ),
            "get_document_registry_status": (
                EmptyInput,
                self._get_document_registry_status,
                "Return indexed documents, chunks, parser status, and classification counts.",
            ),
            "validate_citations": (
                CitationValidationInput,
                self._validate_citations,
                "Check whether the answer cites retrieved evidence identifiers.",
            ),
            "request_human_review": (
                HumanReviewInput,
                self._request_human_review,
                "Create a structured human review request.",
            ),
            "log_feedback": (
                FeedbackInput,
                self._log_feedback,
                "Record answer feedback for evaluation and audit.",
            ),
        }

    def call(self, name: str, payload: dict[str, Any], auth: AuthContext, trace_id: str) -> ToolEnvelope:
        if name not in self._tools:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown tool: {name}")
        input_model, handler, _ = self._tools[name]
        try:
            policy_engine.enforce_tool_call(auth, name)
        except HTTPException:
            UNAUTHORIZED_ATTEMPTS.labels(resource=name).inc()
            raise

        parsed = input_model.model_validate(payload)
        with trace_manager.span(trace_id, "tool_call_started", {"tool": name, "payload": payload}) as span:
            try:
                data = handler(parsed, auth, trace_id)
                data = sanitize_tool_output(data)
                TOOL_COUNT.labels(tool=name, status="ok").inc()
                span.attributes_json = json.dumps({"tool": name, "ok": True, "output_keys": sorted(data.keys())})
                return ToolEnvelope(ok=True, tool=name, data=data)
            except Exception as exc:
                TOOL_COUNT.labels(tool=name, status="error").inc()
                span.status = "error"
                span.error = str(exc)
                return ToolEnvelope(ok=False, tool=name, error=str(exc))

    def manifest(self) -> dict[str, Any]:
        return {
            "protocol": "mcp-style-manifest",
            "server": "defence-agent-tools",
            "tools": [
                ToolSpec(
                    name=name,
                    description=description,
                    input_schema=input_model.model_json_schema(),
                ).model_dump()
                for name, (input_model, _, description) in self._tools.items()
            ],
        }

    def _search_doctrine(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        result = hybrid_retriever.search(
            data["query"],
            auth=auth,
            trace_id=trace_id,
            top_k=data["top_k"],
            filters=data.get("filters") or {},
        )
        return {
            "chunks": [chunk.model_dump() for chunk in result.chunks],
            "degradations": result.degradations,
            "retrieval_trace": result.trace,
        }

    def _compare_versions(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        older = hybrid_retriever.search(
            data["query"],
            auth=auth,
            trace_id=trace_id,
            top_k=data["top_k"],
            filters={"version": data["older_version"], "doc_types": ["handbook", "bulletin", "sop"]},
        )
        newer = hybrid_retriever.search(
            data["query"],
            auth=auth,
            trace_id=trace_id,
            top_k=data["top_k"],
            filters={"version": data["newer_version"], "doc_types": ["handbook", "bulletin", "sop"]},
        )
        return {
            "parallel_steps": [
                f"retrieved_version_{data['older_version']}",
                f"retrieved_version_{data['newer_version']}",
            ],
            "older": [chunk.model_dump() for chunk in older.chunks],
            "newer": [chunk.model_dump() for chunk in newer.chunks],
            "degradations": older.degradations + newer.degradations,
        }

    def _analyze_table_with_python(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        rows = _markdown_table_to_rows(data["table_markdown"])
        code = data.get("code") or DEFAULT_TABLE_ANALYSIS_CODE
        sandbox_result = run_sandboxed_python(
            code=code,
            input_data={"rows": rows, "threshold": data["threshold"]},
            trace_id=trace_id,
            user_id=auth.user_id,
            timeout_seconds=5,
        )
        return {
            "code": code,
            "rows": rows,
            "threshold": data["threshold"],
            "sandbox": sandbox_result,
        }

    def _get_document_registry_status(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        return document_registry_status()

    def _validate_citations(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        cited = citation_ids_from_answer(data["answer"])
        expected = set(data["citation_ids"])
        return {
            "cited": sorted(cited),
            "expected": sorted(expected),
            "valid": bool(cited) and cited.issubset(expected),
            "missing": sorted(expected - cited),
            "unknown": sorted(cited - expected),
        }

    def _request_human_review(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        return {
            "review_id": f"hr_{trace_id[-10:]}",
            "status": "queued",
            "requested_by": auth.user_id,
            "reason": data["reason"],
            "route": data["route"],
            "query": data["query"],
        }

    def _log_feedback(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        with Session(engine) as session:
            feedback = Feedback(
                trace_id=data["trace_id"],
                user_id=auth.user_id,
                helpful=data["helpful"],
                comment=data.get("comment"),
            )
            session.add(feedback)
            session.commit()
            session.refresh(feedback)
        return {"feedback_id": feedback.id, "stored": True}


def _markdown_table_to_rows(markdown: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in markdown.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return []
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, Any]] = []
    for line in lines[2:]:
        values = [cell.strip() for cell in line.strip("|").split("|")]
        row = {header: _coerce(value) for header, value in zip(headers, values)}
        rows.append(row)
    return rows


def _coerce(value: str) -> Any:
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


DEFAULT_TABLE_ANALYSIS_CODE = """
df = pd.DataFrame(input_data["rows"])
threshold = float(input_data["threshold"])
below = df[df["readiness_pct"].astype(float) < threshold].copy()
result = {
    "below_threshold": below[["unit", "readiness_pct", "open_actions"]].to_dict("records"),
    "count": int(len(below)),
    "threshold": threshold,
}
""".strip()


tool_registry = ToolRegistry()
