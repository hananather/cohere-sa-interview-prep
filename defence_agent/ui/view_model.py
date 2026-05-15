"""View-model helpers for the Streamlit Defence Agent UI.

The Streamlit app should render structured audit data. It should not infer
citations, access policy, or trace state by scraping generated answer text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from defence_agent.auth.context import DEMO_USERS
from defence_agent.auth.policy import policy_engine
from defence_agent.session import AgentTurnResult


SCHEMA_VERSION = "defence_agent_ui_v1"


@dataclass(frozen=True)
class UiPersona:
    """Persona exposed in the UI.

    The backend still uses its real clearance personas. The UI keeps the demo
    simple by exposing one unclassified user and one cleared user.
    """

    ui_id: str
    label: str
    backend_persona_id: str
    allowed_access: tuple[str, ...]
    visible_access_label: str
    description: str


UI_PERSONAS: dict[str, UiPersona] = {
    "persona_a": UiPersona(
        ui_id="persona_a",
        label="Persona A: Unclassified User",
        backend_persona_id="clearance_unclassified",
        allowed_access=("unclassified",),
        visible_access_label="unclassified",
        description="Can retrieve and cite unclassified documents only.",
    ),
    "persona_b": UiPersona(
        ui_id="persona_b",
        label="Persona B: Cleared User",
        backend_persona_id="clearance_top_secret",
        allowed_access=("unclassified", "secret", "top_secret"),
        visible_access_label="unclassified + restricted corpus",
        description="Can retrieve and cite unclassified, secret, and top-secret documents.",
    ),
}
DEFAULT_UI_PERSONA_ID = "persona_a"


@dataclass(frozen=True)
class SourceView:
    source_id: str
    doc_id: str = ""
    title: str = ""
    page: str = ""
    section: str = ""
    version: str = ""
    effective_date: str = ""
    access_level: str = ""
    language: str = ""
    chunk_id: str = ""
    source_type: str = ""
    source_format: str = ""
    normalized_format: str = ""
    normalization_method: str = ""
    source_pdf_path: str = ""
    manifest_path: str = ""
    canonical_url: str = ""
    source_url: str = ""
    source_pdf_url: str = ""
    source_docx_url: str = ""
    retrieved_date: str = ""
    source_organization: str = ""
    provenance_note: str = ""
    vector_score: float | None = None
    rerank_score: float | None = None
    sent_to_model: bool = False
    authorized_hit: bool = False
    used_in_answer: bool = False

    def as_row(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page,
            "access_level": self.access_level,
            "language": self.language,
            "used_in_answer": self.used_in_answer,
        }


@dataclass(frozen=True)
class CitationView:
    citation_id: str
    display_index: int
    citation_type: str = ""
    answer_start: int | None = None
    answer_end: int | None = None
    answer_text: str = ""
    source_ids: tuple[str, ...] = ()

    @property
    def marker(self) -> str:
        return f"[{self.display_index}]"


@dataclass(frozen=True)
class EvidencePageView:
    """One UI-safe source page used by one or more citations."""

    page_key: str
    display_index: int
    source: SourceView
    citation_ids: tuple[str, ...]
    source_ids: tuple[str, ...]

    @property
    def label(self) -> str:
        title = self.source.title or self.source.doc_id or "Untitled source"
        page = f"p.{self.source.page}" if self.source.page else "page unknown"
        access = self.source.access_level or "unknown access"
        return f"{title} · {page} · {access}"


@dataclass(frozen=True)
class RuntimeStepView:
    index: int
    title: str
    detail: str
    code: str = ""
    status: str = "success"
    markers: tuple[tuple[str, str], ...] = ()
    metrics: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SearchExecutionView:
    call_index: int
    tool_name: str = "search_documents"
    query: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    authorized_source_count: int = 0
    sources_sent_to_answer_count: int = 0
    excluded_source_count: int = 0
    embedding_backend: str = ""
    rerank_backend: str = ""
    collection: str = ""
    vector_score: float | None = None
    rerank_score: float | None = None
    status: str = "observed"
    answerable: bool | None = None


@dataclass(frozen=True)
class ToolCallView:
    call_index: int
    tool_name: str
    query: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    authorized_source_count: int = 0
    sources_sent_to_answer_count: int = 0
    excluded_source_count: int = 0
    embedding_backend: str = ""
    rerank_backend: str = ""
    collection: str = ""
    vector_score: float | None = None
    rerank_score: float | None = None
    status: str = "observed"

    def as_row(self) -> dict[str, Any]:
        return {
            "call": self.call_index,
            "tool_name": self.tool_name,
            "query": self.query,
            "args": self.arguments,
            "filters": self.filters,
            "authorized_sources": self.authorized_source_count,
            "sent_to_model": self.sources_sent_to_answer_count,
            "excluded_sources": self.excluded_source_count,
            "per_page_vector_match": _format_score(self.vector_score),
            "per_page_rerank_relevance": _format_score(self.rerank_score),
            "status": self.status,
        }


@dataclass(frozen=True)
class DefenceAgentViewModel:
    schema_version: str
    session_id: str
    user_id: str
    ui_persona_id: str
    backend_persona_id: str
    persona_label: str
    allowed_access: tuple[str, ...]
    visible_access_label: str
    query: str
    answer: str
    raw_answer: str
    display_answer: str
    answerability: str
    answerability_reason: str
    citation_mode: str
    citation_validation: dict[str, Any]
    citation_quality: dict[str, Any]
    target_answer_language: str
    documents_sent_to_model: int
    thinking_blocks: tuple[dict[str, Any], ...]
    retrieval_status: str
    citations: tuple[CitationView, ...]
    evidence_pages: tuple[EvidencePageView, ...]
    sources: dict[str, SourceView]
    tool_calls: tuple[ToolCallView, ...]
    runtime_steps: tuple[RuntimeStepView, ...]
    authorized_source_rows: tuple[dict[str, Any], ...]
    excluded_source_summary: tuple[dict[str, Any], ...]
    sanitized_answer_audit: dict[str, Any]
    answer_audit: dict[str, Any]
    run_elapsed_seconds: float | None = None


def persona_for_ui_id(ui_persona_id: str) -> UiPersona:
    """Return a UI persona, falling back closed to the unclassified persona."""

    return UI_PERSONAS.get(ui_persona_id, UI_PERSONAS[DEFAULT_UI_PERSONA_ID])


def ui_persona_for_backend_id(backend_persona_id: str) -> UiPersona:
    """Map backend transcript persona IDs to the closest visible UI persona."""

    if backend_persona_id == "clearance_unclassified":
        return UI_PERSONAS["persona_a"]
    return UI_PERSONAS["persona_b"]


def allowed_access_for_backend_persona(backend_persona_id: str) -> tuple[str, ...]:
    auth = DEMO_USERS[backend_persona_id]
    return tuple(policy_engine.acl_filter(auth).allowed_classifications)


def build_view_model(
    result: AgentTurnResult,
    *,
    ui_persona_id: str | None = None,
    run_elapsed_seconds: float | None = None,
) -> DefenceAgentViewModel:
    """Normalize an ``AgentTurnResult`` for safe, predictable UI rendering."""

    persona = persona_for_ui_id(ui_persona_id or ui_persona_for_backend_id(result.persona_id).ui_id)
    if result.persona_id != persona.backend_persona_id:
        raise ValueError("Backend persona did not match the selected UI persona.")

    audit = result.answer_audit or {}
    retrieval = audit.get("retrieval", {}) if isinstance(audit.get("retrieval"), dict) else {}
    generation = audit.get("generation", {}) if isinstance(audit.get("generation"), dict) else {}

    sources = _build_source_registry(audit)
    citations = _build_citations(audit)
    evidence_pages = _build_evidence_pages(citations, sources)
    search_executions = _build_search_executions(audit)
    tool_calls = _build_tool_calls(search_executions)
    answerability, reason = _answerability(retrieval, result.answer)
    display_answer = _display_answer(result.raw_answer or result.answer)
    documents_sent_to_model = int(result.documents_sent_to_model or generation.get("document_count", 0) or 0)
    runtime_steps = _build_runtime_steps(
        query=str(audit.get("query") or ""),
        persona=persona,
        tool_calls=tool_calls,
        answerability=answerability,
        answerability_reason=reason,
        retrieval_status=result.retrieval_status,
        documents_sent_to_model=documents_sent_to_model,
        citation_count=len(citations),
        evidence_page_count=len(evidence_pages),
        generation_model=str(generation.get("model", "") or ""),
    )

    return DefenceAgentViewModel(
        schema_version=SCHEMA_VERSION,
        session_id=result.session_id,
        user_id=result.user_id,
        ui_persona_id=persona.ui_id,
        backend_persona_id=result.persona_id,
        persona_label=persona.label,
        allowed_access=allowed_access_for_backend_persona(result.persona_id),
        visible_access_label=persona.visible_access_label,
        query=str(audit.get("query") or ""),
        answer=result.answer,
        raw_answer=result.raw_answer,
        display_answer=display_answer,
        answerability=answerability,
        answerability_reason=reason,
        citation_mode=str(result.citation_mode or generation.get("citation_mode", "")),
        citation_validation=dict(result.citation_validation or generation.get("citation_resolution", {}) or {}),
        citation_quality=dict(generation.get("citation_quality", {}) or {}),
        target_answer_language=str(generation.get("target_answer_language", "auto") or "auto"),
        documents_sent_to_model=documents_sent_to_model,
        thinking_blocks=tuple(_thinking_blocks(generation.get("thinking_blocks", result.thinking_blocks))),
        retrieval_status=result.retrieval_status,
        citations=tuple(citations),
        evidence_pages=tuple(evidence_pages),
        sources=sources,
        tool_calls=tuple(tool_calls),
        runtime_steps=tuple(runtime_steps),
        authorized_source_rows=tuple(_source_rows(retrieval.get("authorized_sources", []), sources)),
        excluded_source_summary=tuple(_sanitize_excluded_sources(retrieval.get("excluded_sources", []))),
        sanitized_answer_audit=sanitize_answer_audit(audit),
        answer_audit=audit,
        run_elapsed_seconds=run_elapsed_seconds,
    )


def citation_chip_label(citation: CitationView, sources: dict[str, SourceView]) -> str:
    """Build the compact label used by citation buttons in the Ask tab."""

    span = _compact(citation.answer_text, limit=78)
    if span:
        return f"{citation.marker} {span}"
    primary = sources.get(citation.source_ids[0]) if citation.source_ids else None
    fallback = (primary.title or primary.doc_id) if primary is not None else "unresolved evidence"
    return f"{citation.marker} {_compact(fallback, limit=78)}"


def sanitize_answer_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Return a UI-safe audit object with no denied-source identifiers or paths."""

    retrieval = audit.get("retrieval", {}) if isinstance(audit.get("retrieval"), dict) else {}
    generation = audit.get("generation", {}) if isinstance(audit.get("generation"), dict) else {}
    return {
        "query": audit.get("query", ""),
        "session_id": audit.get("session_id", ""),
        "user_id": audit.get("user_id", ""),
        "persona_id": audit.get("persona_id", ""),
        "tool_calls": list(audit.get("tool_calls", []) or []),
        "tool_call_records": list(audit.get("tool_call_records", []) or []),
        "tool_responses": list(audit.get("tool_responses", []) or []),
        "retrieval_status": audit.get("retrieval_status", ""),
        "retrieval": {
            "search_count": retrieval.get("search_count", 0),
            "searches": list(retrieval.get("searches", []) or []),
            "search_queries": list(retrieval.get("search_queries", []) or []),
            "allowed_access": list(retrieval.get("allowed_access", []) or []),
            "filters_applied": list(retrieval.get("filters_applied", []) or []),
            "policy_decisions": list(retrieval.get("policy_decisions", []) or []),
            "answerability": list(retrieval.get("answerability", []) or []),
            "authorized_source_count": len(retrieval.get("authorized_sources", []) or []),
            "sources_sent_to_answer_count": len(retrieval.get("sources_sent_to_answer", []) or []),
            "excluded_sources": _sanitize_excluded_sources(retrieval.get("excluded_sources", [])),
        },
        "generation": {
            "model": generation.get("model", ""),
            "citation_mode": generation.get("citation_mode", ""),
            "citation_resolution": generation.get("citation_resolution", {}),
            "citation_quality": generation.get("citation_quality", {}),
            "document_count": generation.get("document_count", 0),
            "cohere_document_ids": list(generation.get("cohere_document_ids", []) or []),
            "target_answer_language": generation.get("target_answer_language", "auto"),
            "thinking_blocks": _thinking_blocks(generation.get("thinking_blocks", [])),
            "thinking_block_count": generation.get("thinking_block_count", 0),
        },
        "citations": [_sanitize_citation(citation) for citation in audit.get("citations", []) or []],
    }


def _thinking_blocks(raw_blocks: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if not isinstance(raw_blocks, list):
        return blocks
    for index, block in enumerate(raw_blocks, start=1):
        if not isinstance(block, dict):
            continue
        text = str(block.get("thinking", "") or "")
        if not text:
            continue
        try:
            block_index = int(block.get("index") or index)
        except (TypeError, ValueError):
            block_index = index
        blocks.append(
            {
                "index": block_index,
                "type": "thinking",
                "thinking": text,
            }
        )
    return blocks


def _sanitize_citation(citation: Any) -> dict[str, Any]:
    if not isinstance(citation, dict):
        return {}
    return {
        "type": citation.get("type", ""),
        "start": citation.get("start"),
        "end": citation.get("end"),
        "text": citation.get("text", ""),
        "sources": [_sanitize_citation_source(source) for source in citation.get("sources", []) or []],
    }


def _sanitize_citation_source(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    return {
        "source_id": source.get("source_id", "") or source.get("chunk_id", ""),
        "doc_id": source.get("doc_id", ""),
        "title": source.get("title", ""),
        "page": source.get("page", ""),
        "section": source.get("section", ""),
        "access_level": source.get("access_level", ""),
        "language": source.get("language", ""),
        "version": source.get("version", ""),
        "effective_date": source.get("effective_date", ""),
        "canonical_url": source.get("canonical_url", ""),
        "source_url": source.get("source_url", ""),
        "source_pdf_url": source.get("source_pdf_url", ""),
        "source_docx_url": source.get("source_docx_url", ""),
        "source_organization": source.get("source_organization", ""),
        "retrieved_date": source.get("retrieved_date", ""),
        "provenance_note": source.get("provenance_note", ""),
    }


def _build_source_registry(audit: dict[str, Any]) -> dict[str, SourceView]:
    retrieval = audit.get("retrieval", {}) if isinstance(audit.get("retrieval"), dict) else {}
    registry: dict[str, SourceView] = {}

    for source in retrieval.get("authorized_sources", []) or []:
        if isinstance(source, dict):
            _merge_source(registry, _source_from_dict(source, authorized_hit=True))

    for source in retrieval.get("sources_sent_to_answer", []) or []:
        if isinstance(source, dict):
            _merge_source(registry, _source_from_dict(source, sent_to_model=True))

    for citation in audit.get("citations", []) or []:
        if not isinstance(citation, dict):
            continue
        for source in citation.get("sources", []) or []:
            if isinstance(source, dict):
                _merge_source(
                    registry,
                    _source_from_dict(
                        source,
                        used_in_answer=True,
                        sent_to_model=True,
                    ),
                )
    return registry


def _build_citations(audit: dict[str, Any]) -> list[CitationView]:
    citations: list[CitationView] = []
    for index, citation in enumerate(audit.get("citations", []) or [], start=1):
        if not isinstance(citation, dict):
            continue
        source_ids = tuple(_citation_source_ids(citation.get("sources", [])))
        citations.append(
            CitationView(
                citation_id=f"citation_{index}",
                display_index=index,
                citation_type=str(citation.get("type", "") or ""),
                answer_start=_optional_int(citation.get("start")),
                answer_end=_optional_int(citation.get("end")),
                answer_text=str(citation.get("text", "") or ""),
                source_ids=source_ids,
            )
        )
    return citations


def _build_evidence_pages(
    citations: list[CitationView],
    sources: dict[str, SourceView],
) -> list[EvidencePageView]:
    grouped: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    for citation in citations:
        for source_id in citation.source_ids:
            source = sources.get(source_id)
            if source is None:
                continue
            page_key = _evidence_page_key(source)
            if not page_key:
                continue
            if page_key not in grouped:
                grouped[page_key] = {
                    "source": source,
                    "citation_ids": [],
                    "source_ids": [],
                }
                ordered_keys.append(page_key)
            grouped[page_key]["citation_ids"].append(citation.citation_id)
            grouped[page_key]["source_ids"].append(source_id)

    pages: list[EvidencePageView] = []
    for index, page_key in enumerate(ordered_keys, start=1):
        item = grouped[page_key]
        pages.append(
            EvidencePageView(
                page_key=page_key,
                display_index=index,
                source=item["source"],
                citation_ids=tuple(dict.fromkeys(item["citation_ids"])),
                source_ids=tuple(dict.fromkeys(item["source_ids"])),
            )
        )
    return pages


def _build_runtime_steps(
    *,
    query: str,
    persona: UiPersona,
    tool_calls: list[ToolCallView],
    answerability: str,
    answerability_reason: str,
    retrieval_status: str,
    documents_sent_to_model: int,
    citation_count: int,
    evidence_page_count: int,
    generation_model: str,
) -> list[RuntimeStepView]:
    steps: list[RuntimeStepView] = [
        RuntimeStepView(
            index=1,
            title="User query",
            detail="Captured from the Ask field.",
        ),
        RuntimeStepView(
            index=2,
            title="Agent runtime and session context",
            detail="The agent runtime sends the query and session context to the model runtime.",
            metrics=(("Persona", _persona_runtime_label(persona)),),
        ),
    ]

    if tool_calls:
        plan_detail = f"Command A planned {len(tool_calls)} ordered search_documents call(s)."
        plan_status = "success"
    else:
        plan_detail = (
            "Command A did not record a fresh search_documents call for this turn; "
            "the UI is showing the available audit state."
        )
        plan_status = "neutral"
    steps.append(
        RuntimeStepView(
            index=3,
            title="Model runtime tool plan",
            detail=plan_detail,
            status=plan_status,
            markers=(("Command A", "command"),),
        )
    )

    next_index = 4
    for call in tool_calls:
        metrics = _search_runtime_metrics(call)
        steps.append(
            RuntimeStepView(
                index=next_index,
                title=f"Tool call {call.call_index}: search_documents",
                detail=(
                    "Embed v4 embeds the query; vector search returns authorized page candidates; "
                    "Rerank v4 ranks each candidate page."
                ),
                code=_search_call_code(call),
                metrics=metrics,
            )
        )
        next_index += 1

    if answerability == "REFUSED":
        gate_status = "denied"
        if documents_sent_to_model:
            gate_detail = "Evidence not sufficient after reviewing authorized pages."
        else:
            gate_detail = "Evidence not sufficient before generation."
    else:
        gate_status = "success"
        gate_detail = "Evidence sufficient."
    steps.append(
        RuntimeStepView(
            index=next_index,
            title="Evidence check",
            detail=gate_detail,
            status=gate_status,
        )
    )
    next_index += 1

    if documents_sent_to_model:
        model = generation_model or "command-a-03-2025"
        if answerability == "REFUSED":
            generation_detail = (
                f"Cohere Chat / Command A ({model}) reviewed "
                f"{documents_sent_to_model} authorized page(s) as documents=authorized_pages and returned an insufficiency refusal."
            )
            generation_status = "denied"
        else:
            generation_detail = (
                f"Cohere Chat / Command A ({model}) received "
                f"{documents_sent_to_model} authorized page(s) as documents=authorized_pages."
            )
            generation_status = "success"
    else:
        generation_detail = (
            "Zero-doc generation path: no source text was sent to Command A, "
            "and the UI rendered a refusal."
        )
        generation_status = "denied"
    steps.append(
        RuntimeStepView(
            index=next_index,
            title="Grounded generation",
            detail=generation_detail,
            code="documents=authorized_pages",
            status=generation_status,
            markers=(("Command A", "command"),),
            metrics=(("Documents sent", str(documents_sent_to_model)),),
        )
    )
    next_index += 1

    steps.append(
        RuntimeStepView(
            index=next_index,
            title="UI output",
            detail="Answer and citations rendered; detailed audit stays in Trace.",
            status="denied" if answerability == "REFUSED" else "success",
            metrics=(
                ("Citations", str(citation_count)),
                ("Evidence pages", str(evidence_page_count)),
            ),
        )
    )
    return steps


def _search_runtime_metrics(call: ToolCallView) -> tuple[tuple[str, str], ...]:
    return (("Authorized pages", str(call.authorized_source_count)),)


def _persona_runtime_label(persona: UiPersona) -> str:
    label = persona.label
    label = label.replace("Persona A: ", "Persona A · ")
    label = label.replace("Persona B: ", "Persona B · ")
    return f"{label} · {persona.visible_access_label}"


def _search_call_code(call: ToolCallView) -> str:
    query = call.query or str(call.arguments.get("query", "") or "query not recorded")
    args = [f'query="{query}"']
    if call.arguments.get("top_k") not in (None, ""):
        args.append(f"top_k={call.arguments['top_k']}")
    return f"{call.tool_name}({', '.join(args)})"


def _compact_filter_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _build_search_executions(audit: dict[str, Any]) -> list[SearchExecutionView]:
    retrieval = audit.get("retrieval", {}) if isinstance(audit.get("retrieval"), dict) else {}
    names = [str(name) for name in audit.get("tool_calls", []) or [] if str(name) == "search_documents"]
    records = [
        item
        for item in audit.get("tool_call_records", []) or []
        if isinstance(item, dict) and _is_search_record(item)
    ]
    searches = [item for item in retrieval.get("searches", []) or [] if isinstance(item, dict)]
    queries = [str(query) for query in retrieval.get("search_queries", []) or [] if str(query)]
    filters = [item for item in retrieval.get("filters_applied", []) or [] if isinstance(item, dict)]
    answerability_records = [item for item in retrieval.get("answerability", []) or [] if isinstance(item, dict)]
    tool_cache = retrieval.get("tool_cache", {})
    if not isinstance(tool_cache, dict):
        tool_cache = {}
    cache_events = [
        item
        for item in tool_cache.get("events", [])
        if isinstance(item, dict) and _is_search_cache_event(item)
    ]

    search_count = max(len(names), len(records), len(searches), len(queries), len(cache_events))
    rows: list[SearchExecutionView] = []
    for index in range(1, search_count + 1):
        record = records[index - 1] if index - 1 < len(records) else {}
        raw_args = dict(record.get("args", {}) if isinstance(record.get("args", {}), dict) else {})
        cache_event = cache_events[index - 1] if index - 1 < len(cache_events) else {}
        cache_args = cache_event.get("normalized_args", {}) if isinstance(cache_event.get("normalized_args", {}), dict) else {}
        for key, value in cache_args.items():
            raw_args.setdefault(key, value)
        search = searches[index - 1] if index - 1 < len(searches) else {}
        answerability = answerability_records[index - 1] if index - 1 < len(answerability_records) else {}
        evidence_quality = (
            answerability.get("evidence_quality", {})
            if isinstance(answerability.get("evidence_quality", {}), dict)
            else {}
        )
        query = str(
            search.get("query")
            or raw_args.get("query")
            or (queries[index - 1] if index - 1 < len(queries) else "")
        )
        raw_filters = search.get("filters_applied")
        if not isinstance(raw_filters, dict):
            raw_filters = (
                filters[index - 1]
                if index - 1 < len(filters) and isinstance(filters[index - 1], dict)
                else {}
            )
        raw_filters = _filters_from_args_and_retrieval(raw_filters, raw_args, retrieval)
        rows.append(
            SearchExecutionView(
                call_index=index,
                tool_name=str(record.get("tool_name") or cache_event.get("tool_name") or "search_documents"),
                query=query,
                arguments=dict(raw_args),
                filters=raw_filters,
                authorized_source_count=int(
                    search.get("authorized_source_count")
                    or evidence_quality.get("selected_source_count")
                    or 0
                ),
                sources_sent_to_answer_count=int(search.get("sources_sent_to_answer_count", 0) or 0),
                excluded_source_count=int(search.get("excluded_source_count", 0) or 0),
                embedding_backend=str(search.get("embedding_backend", "") or ""),
                rerank_backend=str(search.get("rerank_backend", "") or ""),
                collection=str(search.get("collection", "") or ""),
                vector_score=_optional_float(
                    search.get("top_authorized_vector_score")
                    if search.get("top_authorized_vector_score") is not None
                    else evidence_quality.get("top_authorized_vector_score")
                ),
                rerank_score=_optional_float(
                    search.get("top_authorized_rerank_score")
                    if search.get("top_authorized_rerank_score") is not None
                    else evidence_quality.get("top_authorized_rerank_score")
                ),
                status=str(search.get("policy_decision") or answerability.get("reason") or "observed"),
                answerable=(
                    answerability.get("answerable")
                    if isinstance(answerability.get("answerable"), bool)
                    else None
                ),
            )
        )
    return rows


def _build_tool_calls(search_executions: list[SearchExecutionView]) -> list[ToolCallView]:
    return [
        ToolCallView(
            call_index=search.call_index,
            tool_name=search.tool_name,
            query=search.query,
            arguments=search.arguments,
            filters=search.filters,
            authorized_source_count=search.authorized_source_count,
            sources_sent_to_answer_count=search.sources_sent_to_answer_count,
            excluded_source_count=search.excluded_source_count,
            embedding_backend=search.embedding_backend,
            rerank_backend=search.rerank_backend,
            collection=search.collection,
            vector_score=search.vector_score,
            rerank_score=search.rerank_score,
            status=search.status,
        )
        for search in search_executions
    ]


def _is_search_record(record: dict[str, Any]) -> bool:
    tool_name = str(record.get("tool_name", "") or "")
    args = record.get("args", {}) if isinstance(record.get("args", {}), dict) else {}
    return tool_name == "search_documents" or (not tool_name and "query" in args)


def _is_search_cache_event(event: dict[str, Any]) -> bool:
    tool_name = str(event.get("tool_name", "") or "")
    args = event.get("normalized_args", {}) if isinstance(event.get("normalized_args", {}), dict) else {}
    return tool_name == "search_documents" or (not tool_name and "query" in args)


def _filters_from_args_and_retrieval(
    filters: dict[str, Any],
    args: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(filters)
    if "status_filter" in args and "status" not in merged:
        merged["status"] = args["status_filter"]
    if "language" in args and "language" not in merged:
        merged["language"] = args["language"]
    allowed_access = retrieval.get("allowed_access")
    if allowed_access and "access_level" not in merged:
        merged["access_level"] = list(allowed_access)
    return merged


def _source_from_dict(
    source: dict[str, Any],
    *,
    sent_to_model: bool = False,
    authorized_hit: bool = False,
    used_in_answer: bool = False,
) -> SourceView:
    source_id = _source_key(source)
    return SourceView(
        source_id=source_id,
        doc_id=str(source.get("doc_id", "") or ""),
        title=str(source.get("title", "") or ""),
        page=str(source.get("page", "") or ""),
        section=str(source.get("section", "") or ""),
        version=str(source.get("version", "") or ""),
        effective_date=str(source.get("effective_date", "") or ""),
        access_level=str(source.get("access_level", "") or ""),
        language=str(source.get("language", "") or ""),
        chunk_id=str(source.get("chunk_id") or source_id),
        source_type=str(source.get("source_type", "") or ""),
        source_format=str(source.get("source_format", "") or ""),
        normalized_format=str(source.get("normalized_format", "") or ""),
        normalization_method=str(source.get("normalization_method", "") or ""),
        source_pdf_path=str(source.get("source_pdf_path", "") or ""),
        manifest_path=str(source.get("manifest_path", "") or ""),
        canonical_url=str(source.get("canonical_url", "") or ""),
        source_url=str(source.get("source_url", "") or ""),
        source_pdf_url=str(source.get("source_pdf_url", "") or ""),
        source_docx_url=str(source.get("source_docx_url", "") or ""),
        retrieved_date=str(source.get("retrieved_date", "") or ""),
        source_organization=str(source.get("source_organization", "") or ""),
        provenance_note=str(source.get("provenance_note", "") or ""),
        vector_score=_optional_float(source.get("vector_score")),
        rerank_score=_optional_float(source.get("rerank_score")),
        sent_to_model=sent_to_model,
        authorized_hit=authorized_hit,
        used_in_answer=used_in_answer,
    )


def _merge_source(registry: dict[str, SourceView], incoming: SourceView) -> None:
    if not incoming.source_id:
        return
    existing = registry.get(incoming.source_id)
    if existing is None:
        registry[incoming.source_id] = incoming
        return

    registry[incoming.source_id] = SourceView(
        source_id=existing.source_id,
        doc_id=_prefer(existing.doc_id, incoming.doc_id),
        title=_prefer(existing.title, incoming.title),
        page=_prefer(existing.page, incoming.page),
        section=_prefer(existing.section, incoming.section),
        version=_prefer(existing.version, incoming.version),
        effective_date=_prefer(existing.effective_date, incoming.effective_date),
        access_level=_prefer(existing.access_level, incoming.access_level),
        language=_prefer(existing.language, incoming.language),
        chunk_id=_prefer(existing.chunk_id, incoming.chunk_id),
        source_type=_prefer(existing.source_type, incoming.source_type),
        source_format=_prefer(existing.source_format, incoming.source_format),
        normalized_format=_prefer(existing.normalized_format, incoming.normalized_format),
        normalization_method=_prefer(existing.normalization_method, incoming.normalization_method),
        source_pdf_path=_prefer(existing.source_pdf_path, incoming.source_pdf_path),
        manifest_path=_prefer(existing.manifest_path, incoming.manifest_path),
        canonical_url=_prefer(existing.canonical_url, incoming.canonical_url),
        source_url=_prefer(existing.source_url, incoming.source_url),
        source_pdf_url=_prefer(existing.source_pdf_url, incoming.source_pdf_url),
        source_docx_url=_prefer(existing.source_docx_url, incoming.source_docx_url),
        retrieved_date=_prefer(existing.retrieved_date, incoming.retrieved_date),
        source_organization=_prefer(existing.source_organization, incoming.source_organization),
        provenance_note=_prefer(existing.provenance_note, incoming.provenance_note),
        vector_score=existing.vector_score if existing.vector_score is not None else incoming.vector_score,
        rerank_score=existing.rerank_score if existing.rerank_score is not None else incoming.rerank_score,
        sent_to_model=existing.sent_to_model or incoming.sent_to_model,
        authorized_hit=existing.authorized_hit or incoming.authorized_hit,
        used_in_answer=existing.used_in_answer or incoming.used_in_answer,
    )


def _citation_source_ids(raw_sources: Any) -> list[str]:
    source_ids: list[str] = []
    for source in raw_sources or []:
        if not isinstance(source, dict):
            continue
        source_id = _source_key(source)
        if source_id:
            source_ids.append(source_id)
    return source_ids


def _source_key(source: dict[str, Any]) -> str:
    source_id = str(
        source.get("source_id")
        or source.get("chunk_id")
        or source.get("cohere_document_id")
        or source.get("citation_id")
        or ""
    ).strip()
    if source_id:
        return source_id
    doc_id = str(source.get("doc_id", "") or "").strip()
    page = str(source.get("page", "") or "").strip()
    return f"{doc_id}:page:{page}" if doc_id and page else doc_id


def _evidence_page_key(source: SourceView) -> str:
    doc_id = source.doc_id.strip()
    page = source.page.strip()
    access = source.access_level.strip()
    if not doc_id:
        return ""
    return f"{doc_id}:page:{page or 'unknown'}:access:{access or 'unknown'}"


def _source_rows(raw_sources: Any, sources: dict[str, SourceView]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_source in raw_sources or []:
        if not isinstance(raw_source, dict):
            continue
        key = _source_key(raw_source)
        source = sources.get(key) or _source_from_dict(raw_source)
        if source.source_id in seen:
            continue
        seen.add(source.source_id)
        rows.append(source.as_row())
    return rows


def _sanitize_excluded_sources(excluded_sources: Any) -> list[dict[str, Any]]:
    summary: dict[tuple[str, str], int] = {}
    for source in excluded_sources or []:
        if not isinstance(source, dict):
            continue
        access_level = str(source.get("access_level", "restricted") or "restricted")
        reason = str(source.get("reason", "access_denied") or "access_denied")
        summary[(access_level, reason)] = summary.get((access_level, reason), 0) + 1
    return [
        {"access_level": access_level, "exclusion_reason": reason, "count": count}
        for (access_level, reason), count in sorted(summary.items())
    ]


def _answerability(retrieval: dict[str, Any], answer: str) -> tuple[str, str]:
    records = [item for item in retrieval.get("answerability", []) or [] if isinstance(item, dict)]
    if records:
        if any(item.get("answerable") is False for item in records):
            reason = next(str(item.get("reason", "")) for item in records if item.get("answerable") is False)
            return "REFUSED", reason or "insufficient_authorized_evidence"
        if any(item.get("answerable") is True for item in records):
            return "ANSWERED", "authorized_evidence_sent"

    if retrieval.get("sources_sent_to_answer"):
        return "ANSWERED", "authorized_evidence_sent"
    text = answer.lower()
    if any(
        phrase in text
        for phrase in ("do not have enough", "evidence is insufficient", "insufficient evidence", "cannot answer")
    ):
        return "REFUSED", "insufficient_authorized_evidence"
    return "PARTIAL", "answerability_not_explicit"


def _prefer(current: str, incoming: str) -> str:
    return current or incoming


def _display_answer(answer: str) -> str:
    """Remove backend-only citation markers from the answer shown in Ask."""

    text = str(answer or "").strip()
    text = re.sub(r"\s*\[(?:C\d+(?:,\s*C\d+)*)\]", "", text)
    raw_lines = [line for line in text.splitlines() if line.strip()]
    bullet_like_count = sum(1 for line in raw_lines if re.match(r"^\s*[-*]\s+", line))
    lines = [
        re.sub(r"[ \t]{2,}", " ", re.sub(r"^\s*[-*]\s+", "", line)).strip()
        for line in raw_lines
    ]
    lines = [line for line in lines if line]
    if bullet_like_count >= 2:
        return " ".join(lines)
    return "\n\n".join(lines)


def _compact(text: str, *, limit: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "..."


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_score(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"
