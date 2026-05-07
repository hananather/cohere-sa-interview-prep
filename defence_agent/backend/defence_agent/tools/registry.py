from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from typing import Any, Callable

from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from defence_agent.auth.context import AuthContext
from defence_agent.auth.policy import policy_engine
from defence_agent.config import get_settings
from defence_agent.db import engine
from defence_agent.ingestion.indexer import TABLE_DOCUMENT_METADATA, document_registry_status
from defence_agent.models import Chunk, Feedback
from defence_agent.observability.metrics import TOOL_COUNT, UNAUTHORIZED_ATTEMPTS
from defence_agent.observability.plugins import plugin_manager
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


class DocumentSectionsInput(BaseModel):
    doc_id: str
    section_ids: list[str] = Field(default_factory=list)


class FollowReferencesInput(BaseModel):
    query: str
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=8)


class CompareVersionsInput(BaseModel):
    query: str
    older_version: str = "2024"
    newer_version: str = "2025"
    doc_family: str | None = None
    version_a: str | None = None
    version_b: str | None = None
    focus: str | None = None
    top_k: int = Field(default=4, ge=1, le=8)


class AnalyzeTableInput(BaseModel):
    table_markdown: str
    threshold: float = 80
    code: str | None = None


class GetTableInput(BaseModel):
    table_name: str
    filters: dict[str, Any] = Field(default_factory=dict)


class RunTableAnalysisInput(BaseModel):
    dataset_id: str
    analysis_goal: str
    rows: list[dict[str, Any]]
    today: str = "2026-05-06"
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
    helpful: bool | None = None
    rating: str | None = None
    query_id: str | None = None
    user_query: str | None = None
    answer: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    route: str | None = None
    tools_called: list[str] = Field(default_factory=list)
    reason: str | None = None
    selected_failure_type: str | None = None
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
            "search_documents": (
                SearchDoctrineInput,
                self._search_doctrine,
                "Search authorized document chunks with metadata filters and reranking.",
            ),
            "get_document_sections": (
                DocumentSectionsInput,
                self._get_document_sections,
                "Retrieve full sections from an already-authorized document.",
            ),
            "follow_references": (
                FollowReferencesInput,
                self._follow_references,
                "Extract referenced documents or sections from retrieved chunks and search them.",
            ),
            "compare_versions": (
                CompareVersionsInput,
                self._compare_versions,
                "Compare authorized chunks across two document versions.",
            ),
            "compare_document_versions": (
                CompareVersionsInput,
                self._compare_versions,
                "Compare two versions from a document family using authorized chunks.",
            ),
            "analyze_table_with_python": (
                AnalyzeTableInput,
                self._analyze_table_with_python,
                "Run sandboxed Python against authorized table data.",
            ),
            "get_table": (
                GetTableInput,
                self._get_table,
                "Load an authorized structured table as row dictionaries.",
            ),
            "run_table_analysis": (
                RunTableAnalysisInput,
                self._run_table_analysis,
                "Run controlled Python analysis over an already-authorized table dataset.",
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
            "admin_reindex": (
                EmptyInput,
                self._admin_reindex,
                "Admin-only corpus reindex control. Disabled as a model tool in the demo UI.",
            ),
        }

    def call(self, name: str, payload: dict[str, Any], auth: AuthContext, trace_id: str) -> ToolEnvelope:
        if name not in self._tools:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown tool: {name}")
        input_model, handler, _ = self._tools[name]
        try:
            plugin_manager.before_tool(trace_id, name, payload, auth)
            policy_engine.enforce_tool_call(auth, name)
        except HTTPException:
            UNAUTHORIZED_ATTEMPTS.labels(resource=name).inc()
            raise

        parsed = input_model.model_validate(payload)
        with trace_manager.span(trace_id, "tool_call_started", {"tool": name, "payload": payload}) as span:
            try:
                data = handler(parsed, auth, trace_id)
                data = sanitize_tool_output(data)
                plugin_manager.after_tool(trace_id, name, data, auth)
                TOOL_COUNT.labels(tool=name, status="ok").inc()
                span.attributes_json = json.dumps({"tool": name, "ok": True, "output_keys": sorted(data.keys())})
                return ToolEnvelope(ok=True, tool=name, data=data)
            except Exception as exc:
                plugin_manager.on_tool_error(trace_id, name, str(exc), auth)
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

    def _get_document_sections(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        with Session(engine) as session:
            query = select(Chunk).where(Chunk.document_id == data["doc_id"]).order_by(Chunk.chunk_index)
            chunks = session.exec(query).all()
        authorized = [chunk for chunk in chunks if policy_engine.can_access_chunk(auth, chunk)]
        if data["section_ids"]:
            wanted = set(data["section_ids"])
            authorized = [chunk for chunk in authorized if chunk.row_id in wanted or chunk.section in wanted]
        return {"chunks": [_chunk_to_dict(chunk) for chunk in authorized]}

    def _follow_references(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        text_blob = " ".join(str(chunk.get("text", "")) for chunk in data.get("retrieved_chunks", []))
        references = sorted(set(re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", text_blob)))
        chunks: list[dict[str, Any]] = []
        for reference in references:
            result = hybrid_retriever.search(
                f"{reference} {data['query']}",
                auth=auth,
                trace_id=trace_id,
                top_k=data["top_k"],
                filters={"doc_ids": [reference], "status": "approved"},
            )
            chunks.extend(chunk.model_dump() for chunk in result.chunks)
        return {"references": references, "chunks": chunks}

    def _compare_versions(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        older_version = data.get("version_a") or data["older_version"]
        newer_version = data.get("version_b") or data["newer_version"]
        filters_base = {"doc_types": ["planning_brief_approval", "handbook", "bulletin", "sop"]}
        if data.get("doc_family"):
            filters_base = {"doc_family": data["doc_family"]}
        older = hybrid_retriever.search(
            data["query"],
            auth=auth,
            trace_id=trace_id,
            top_k=data["top_k"],
            filters={**filters_base, "version": older_version},
        )
        newer = hybrid_retriever.search(
            data["query"],
            auth=auth,
            trace_id=trace_id,
            top_k=data["top_k"],
            filters={**filters_base, "version": newer_version},
        )
        return {
            "parallel_steps": [
                f"retrieved_version_{older_version}",
                f"retrieved_version_{newer_version}",
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

    def _get_table(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        table_name = data["table_name"]
        if table_name not in TABLE_DOCUMENT_METADATA:
            raise ValueError(f"Unknown table: {data['table_name']}")
        with Session(engine) as session:
            chunks = session.exec(
                select(Chunk)
                .where(Chunk.document_id == table_name)
                .order_by(Chunk.chunk_index)
            ).all()
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            if not policy_engine.can_access_chunk(auth, chunk):
                continue
            row = _table_chunk_to_row(chunk)
            if _row_matches(row, data.get("filters") or {}):
                rows.append(row)
        table_metadata = TABLE_DOCUMENT_METADATA[table_name]
        return {
            "dataset_id": table_name,
            "row_count": len(rows),
            "rows": rows,
            "source": {
                "document_id": table_name,
                "title": table_metadata["title"],
                "access": "authorized_rows_only",
            },
        }

    def _run_table_analysis(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        data = payload.model_dump()
        code = data.get("code") or _analysis_code_for_goal(data["analysis_goal"])
        sandbox_result = run_sandboxed_python(
            code=code,
            input_data={"rows": data["rows"], "today": data["today"], "analysis_goal": data["analysis_goal"]},
            trace_id=trace_id,
            user_id=auth.user_id,
            timeout_seconds=5,
        )
        trace_manager.add_span(
            trace_id,
            "sandbox_policy_checked",
            {
                "policy": sandbox_result.get("policy"),
                "dataset_id": data["dataset_id"],
                "row_count": len(data["rows"]),
                "row_ids": sandbox_result.get("row_ids", []),
            },
        )
        return {
            "dataset_id": data["dataset_id"],
            "analysis_goal": data["analysis_goal"],
            "code": code,
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
        helpful = data.get("helpful")
        if helpful is None:
            helpful = data.get("rating") == "yes"
        with Session(engine) as session:
            feedback = Feedback(
                trace_id=data["trace_id"],
                user_id=auth.user_id,
                helpful=bool(helpful),
                comment=data.get("comment") or data.get("reason"),
            )
            session.add(feedback)
            session.commit()
            session.refresh(feedback)
        feedback_dir = get_settings().data_dir / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "trace_id": data["trace_id"],
            "query_id": data.get("query_id"),
            "user_query": data.get("user_query"),
            "answer": data.get("answer"),
            "citations": data.get("citations") or [],
            "route": data.get("route"),
            "tools_called": data.get("tools_called") or [],
            "rating": data.get("rating") or ("yes" if helpful else "no"),
            "helpful": bool(helpful),
            "reason": data.get("reason") or data.get("comment"),
            "selected_failure_type": data.get("selected_failure_type"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": auth.user_id,
        }
        with (feedback_dir / "feedback_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return {"feedback_id": feedback.id, "stored": True}

    def _admin_reindex(self, payload: BaseModel, auth: AuthContext, trace_id: str) -> dict[str, Any]:
        return {
            "status": "disabled_in_demo_tool_registry",
            "message": "Use the explicit admin API endpoint for reindexing. This registry entry exists to demonstrate admin tool governance.",
        }


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


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "section": chunk.section,
        "page": chunk.page,
        "text": chunk.text,
        "summary": chunk.summary,
        "classification": chunk.classification,
        "version": chunk.version,
        "effective_date": chunk.effective_date,
        "status": chunk.status,
        "doc_family": chunk.doc_family,
        "owner": chunk.owner,
        "review_due": chunk.review_due,
        "language": chunk.language,
        "source_type": chunk.source_type,
        "row_id": chunk.row_id,
        "doc_type": chunk.doc_type,
        "source_uri": chunk.source_uri,
    }


def _tracker_text_to_row(chunk: Chunk) -> dict[str, Any]:
    text = chunk.text
    match = re.search(
        r"Row (?P<row_id>R\d+): (?P<doc_id>[^ ]+) is (?P<title>.+?) owned by (?P<owner>.+?)\. Status (?P<status>.+?)\. Effective date (?P<effective_date>.+?)\. Next review due (?P<next_review_due>.+?)\. Access level (?P<access_level>.+?)\. Language (?P<language>.+?)\.",
        text,
    )
    if not match:
        return {
            "row_id": chunk.row_id,
            "doc_id": chunk.document_id,
            "title": chunk.title,
            "owner": chunk.owner,
            "status": chunk.status,
            "effective_date": chunk.effective_date,
            "next_review_due": chunk.review_due,
            "access_level": chunk.classification,
            "language": chunk.language,
        }
    row = match.groupdict()
    row["doc_family"] = chunk.doc_family
    return row


def _table_chunk_to_row(chunk: Chunk) -> dict[str, Any]:
    if chunk.document_id == "doctrine_review_tracker":
        row = _tracker_text_to_row(chunk)
    else:
        pairs = re.findall(r"([A-Za-z0-9_]+)=([^.]*)\.", chunk.text)
        row = {key: _coerce(value.strip()) for key, value in pairs}
        row.setdefault("row_id", chunk.row_id)
        row.setdefault("doc_id", chunk.document_id)
        row.setdefault("title", chunk.title)
        row.setdefault("owner", chunk.owner)
        row.setdefault("status", chunk.status)
        row.setdefault("effective_date", chunk.effective_date)
        row.setdefault("next_review_due", chunk.review_due)
        row.setdefault("access_level", chunk.classification)
        row.setdefault("language", chunk.language)
        row.setdefault("doc_family", chunk.doc_family)
    row["row_id"] = chunk.row_id
    row["source_table"] = chunk.document_id
    return row


def _row_matches(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if isinstance(expected, list):
            if row.get(key) not in expected:
                return False
        elif row.get(key) != expected:
            return False
    return True


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


DEFAULT_OVERDUE_ANALYSIS_CODE = """
df = pd.DataFrame(input_data["rows"])
today = pd.to_datetime(input_data["today"])
df["next_review_due_dt"] = pd.to_datetime(df["next_review_due"])
approved = df[df["status"] == "approved"].copy()
overdue = approved[approved["next_review_due_dt"] < today].copy()
overdue["days_overdue"] = (today - overdue["next_review_due_dt"]).dt.days.astype(int)
grouped = (
    overdue.groupby("owner")
    .agg(overdue_documents=("doc_id", "count"), max_days_overdue=("days_overdue", "max"))
    .reset_index()
    .sort_values(["max_days_overdue", "owner"], ascending=[False, True])
)
result = {
    "today": input_data["today"],
    "rows": overdue[["row_id", "doc_id", "title", "owner", "next_review_due", "days_overdue"]].sort_values("days_overdue", ascending=False).to_dict("records"),
    "grouped": grouped.to_dict("records"),
    "row_ids": overdue["row_id"].tolist(),
}
""".strip()


DEFAULT_COUNT_APPROVED_BY_OWNER_CODE = """
df = pd.DataFrame(input_data["rows"])
approved = df[df["status"] == "approved"].copy()
grouped = (
    approved.groupby("owner")
    .agg(approved_documents=("doc_id", "count"))
    .reset_index()
    .sort_values(["approved_documents", "owner"], ascending=[False, True])
)
result = {
    "analysis_type": "count_approved_by_owner",
    "today": input_data["today"],
    "rows": approved[["row_id", "doc_id", "title", "owner", "status"]].to_dict("records"),
    "grouped": grouped.to_dict("records"),
    "row_ids": approved["row_id"].tolist(),
}
""".strip()


DEFAULT_READINESS_ANALYSIS_CODE = """
df = pd.DataFrame(input_data["rows"])
df["readiness_percent"] = df["readiness_percent"].astype(float)
df["threshold_percent"] = df["threshold_percent"].astype(float)
below = df[df["readiness_percent"] < df["threshold_percent"]].copy()
below["points_below_threshold"] = (below["threshold_percent"] - below["readiness_percent"]).astype(int)
grouped = (
    below.groupby("owner")
    .agg(units_below_threshold=("unit_id", "count"), max_points_below=("points_below_threshold", "max"))
    .reset_index()
    .sort_values(["max_points_below", "owner"], ascending=[False, True])
)
result = {
    "analysis_type": "readiness_below_threshold",
    "today": input_data["today"],
    "rows": below[["row_id", "unit_id", "unit_name", "owner", "readiness_percent", "threshold_percent", "points_below_threshold", "linked_doc_id"]].to_dict("records"),
    "grouped": grouped.to_dict("records"),
    "row_ids": below["row_id"].tolist(),
}
""".strip()


DEFAULT_TABLE_ROWS_CODE = """
df = pd.DataFrame(input_data["rows"])
result = {
    "analysis_type": "table_rows",
    "today": input_data["today"],
    "rows": df.to_dict("records"),
    "row_ids": df["row_id"].tolist() if "row_id" in df.columns else [],
}
""".strip()


def _analysis_code_for_goal(goal: str) -> str:
    lowered = goal.lower()
    if "readiness" in lowered or "below threshold" in lowered or "threshold" in lowered:
        return DEFAULT_READINESS_ANALYSIS_CODE
    if "count approved" in lowered and "owner" in lowered:
        return DEFAULT_COUNT_APPROVED_BY_OWNER_CODE
    if "approval" in lowered or "corrective" in lowered or "annex inventory" in lowered or "annexes" in lowered:
        return DEFAULT_TABLE_ROWS_CODE
    return DEFAULT_OVERDUE_ANALYSIS_CODE


tool_registry = ToolRegistry()
