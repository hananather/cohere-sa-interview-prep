from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from defence_agent.evals.schemas import EvalCase


def tool_names(response: dict[str, Any]) -> list[str]:
    return [
        str(call.get("tool"))
        for call in response.get("tool_calls", [])
        if isinstance(call, dict) and call.get("tool")
    ]


def source_doc_ids(response: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for source in response.get("sources", []):
        doc_id = source.get("document_id")
        if doc_id:
            ids.append(str(doc_id))
    return ids


def citation_doc_ids(response: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for citation in response.get("citations", []):
        doc_id = citation.get("document_id")
        if doc_id:
            ids.append(str(doc_id))
    return ids


def lower_answer(response: dict[str, Any]) -> str:
    return str(response.get("answer", "")).lower()


def expected_doc_ids(case: EvalCase) -> list[str]:
    return [source.doc_id for source in case.expected_sources if source.required]


def disallowed_doc_ids(case: EvalCase) -> list[str]:
    return [source.doc_id for source in case.disallowed_sources]


def contains_all(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return all(str(term).lower() in lowered for term in terms)


def contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(str(term).lower() in lowered for term in terms)


def pass_fail(ok: bool, category: str, **extra: Any) -> dict[str, Any]:
    return {"passed": ok, "failure_category": None if ok else category, **extra}
