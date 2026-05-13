"""Final answer grounding and citation extraction.

ADK owns the agent loop and tool calls. Cohere owns the final grounded answer
because Cohere Chat can return native citation spans from the exact documents
sent to the model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from defence_agent.cohere_gateway import cohere_gateway
from defence_agent.config import get_settings

logger = logging.getLogger(__name__)
# Cohere RAG citations default to accurate mode. The hosted Command A endpoint
# currently rejects an explicit citation_options mode for document RAG calls.
COHERE_CITATION_MODE = "accurate_default"
MAX_CITATION_REPAIR_ATTEMPTS = 1
MAX_GROUNDING_SOURCES = 32


@dataclass(frozen=True)
class GroundedAnswer:
    """Final answer plus citation metadata in a presentation-friendly shape."""

    answer: str
    raw_answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    citation_mode: str = "none"
    citation_validation: dict[str, Any] = field(default_factory=dict)
    documents_sent: int = 0
    document_ids: list[str] = field(default_factory=list)
    model: str = ""


def finalize_answer(
    *,
    query: str,
    sources: list[dict[str, Any]],
    prior_answer: str = "",
    fallback_answer: str = "I do not have enough authorized evidence to answer.",
    target_answer_language: str = "auto",
) -> GroundedAnswer:
    """Return a cited final answer from authorized evidence.

    This makes a direct Cohere Chat call with ``documents=`` and extracts
    ``response.message.citations``.
    """

    clean_sources = _dedupe_sources(sources)
    if not clean_sources:
        return GroundedAnswer(
            answer=fallback_answer,
            raw_answer=fallback_answer,
            citation_mode="none",
            citation_validation={"passed": False, "errors": ["no_authorized_sources"]},
            model="none",
        )

    return _cohere_native_answer(
        query=query,
        prior_answer=prior_answer,
        sources=clean_sources,
        chat_model=get_settings().cohere_chat_model,
        target_answer_language=target_answer_language,
    )


def _cohere_native_answer(
    *,
    query: str,
    prior_answer: str,
    sources: list[dict[str, Any]],
    chat_model: str,
    target_answer_language: str,
) -> GroundedAnswer:
    documents, evidence_by_label = _cohere_documents(sources)
    logger.info(
        "generate_grounded_answer_intent",
        extra={
            "model": chat_model,
            "document_count": len(documents),
            "source_ids": [doc["id"] for doc in documents],
        },
    )
    raw_answer, raw_citations = _cohere_chat(
        chat_model=chat_model,
        query=query,
        prior_answer=prior_answer,
        documents=documents,
        target_answer_language=target_answer_language,
    )
    citations = _normalize_citations(raw_citations, evidence_by_label)
    validation = _validate_native_citations(citations, evidence_by_label, answer=raw_answer, query=query)
    for _ in range(MAX_CITATION_REPAIR_ATTEMPTS):
        if validation.get("passed") and not validation.get("warnings"):
            break
        retry_answer, retry_raw_citations = _cohere_chat(
            chat_model=chat_model,
            query=query,
            prior_answer=prior_answer,
            documents=documents,
            target_answer_language=target_answer_language,
            citation_repair_feedback=_citation_repair_feedback(validation),
        )
        retry_citations = _normalize_citations(retry_raw_citations, evidence_by_label)
        retry_validation = _validate_native_citations(
            retry_citations,
            evidence_by_label,
            answer=retry_answer,
            query=query,
        )
        if _validation_score(retry_validation) >= _validation_score(validation):
            raw_answer = retry_answer
            citations = retry_citations
            validation = retry_validation

    answer = _insert_inline_markers(raw_answer, citations)
    citation_mode = f"cohere_native_{COHERE_CITATION_MODE}"
    if not validation.get("passed"):
        answer = "I do not have enough cited evidence from the authorized documents to answer this question."
        citation_mode = f"{citation_mode}_citation_validation_failed"
    logger.info(
        "generate_grounded_answer_outcome",
        extra={
            "model": chat_model,
            "citation_count": len(citations),
            "validation_passed": validation.get("passed"),
        },
    )
    return GroundedAnswer(
        answer=answer,
        raw_answer=raw_answer,
        citations=citations,
        citation_mode=citation_mode,
        citation_validation=validation,
        documents_sent=len(documents),
        document_ids=[str(doc["id"]) for doc in documents],
        model=chat_model,
    )


def _cohere_chat(
    *,
    chat_model: str,
    query: str,
    prior_answer: str,
    documents: list[dict[str, Any]],
    target_answer_language: str,
    citation_repair_feedback: str = "",
) -> tuple[str, list[Any]]:
    response = cohere_gateway.chat(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": _grounding_system_message(
                    citation_repair_feedback,
                    target_answer_language=target_answer_language,
                ),
            },
            {"role": "user", "content": _grounded_user_message(query, prior_answer)},
        ],
        documents=documents,
        temperature=0.05,
        max_tokens=1000,
    )
    return _message_text(response), _response_citations(response)


def _grounding_system_message(
    citation_repair_feedback: str = "",
    *,
    target_answer_language: str = "auto",
) -> str:
    language_instruction = _target_language_instruction(target_answer_language)
    message = (
        "You are the DefTech Doctrine Intelligence Assistant. "
        "Answer only from the provided authorized documents. "
        f"{language_instruction} "
        "Write a staff-ready answer of about 140 to 220 words unless the evidence is insufficient. "
        "Use one substantial paragraph or two compact paragraphs. "
        "Every factual sentence must be supported by at least one citation. "
        "If a sentence cannot be cited from the provided documents, omit it. "
        "Do not write uncited introductions, transitions, summaries, or conclusions. "
        "Do not emit XML, HTML, <co> tags, footnotes, or custom citation markup. "
        "Return plain text only; the application will render citation markers. "
        "For comparisons, cite each document that supports the comparison. "
        "When English and French sources both support the answer, cite both source languages where natural. "
        "If the provided documents do not support the answer, say that evidence is insufficient in one short sentence."
    )
    if citation_repair_feedback:
        message += "\n\nCitation repair instruction: " + citation_repair_feedback
    return message


def _target_language_instruction(target_answer_language: str) -> str:
    normalized = str(target_answer_language or "auto").strip().lower()
    if normalized == "fr":
        return (
            "Answer in French. Source documents may be English or French; cite the source pages even when "
            "you translate their evidence into French."
        )
    if normalized == "en":
        return (
            "Answer in English. Source documents may be English or French; cite the source pages even when "
            "you translate their evidence into English."
        )
    return (
        "Answer in the same language as the user's question. Source documents may be English or French; "
        "cite the source pages even when source language differs from answer language."
    )


def _grounded_user_message(query: str, prior_answer: str) -> str:
    if not prior_answer:
        return query
    return (
        "Use this previous answer only to resolve references in the follow-up. "
        "Do not treat it as evidence.\n\n"
        f"Previous answer:\n{prior_answer[:1800]}\n\n"
        f"User question:\n{query}"
    )


def _cohere_documents(sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    evidence_by_label: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources, start=1):
        label = f"C{index}"
        stable_id = _cohere_document_id(source, index)
        evidence = dict(source)
        evidence["citation_id"] = label
        evidence["citation"] = f"[{label}]"
        evidence["cohere_document_id"] = stable_id
        evidence_by_label[stable_id] = evidence
        documents.append(
            {
                "id": stable_id,
                "data": {
                    "text": str(source.get("text", "")),
                    "doc_id": str(source.get("doc_id", "")),
                    "title": str(source.get("title", "")),
                    "section": str(source.get("section", "")),
                    "page": source.get("page", ""),
                    "status": str(source.get("status", "")),
                    "version": str(source.get("version", "")),
                    "access_level": str(source.get("access_level", "")),
                    "language": str(source.get("language", "")),
                    "source_type": str(source.get("source_type", "")),
                    "source_format": str(source.get("source_format", "")),
                    "normalized_format": str(source.get("normalized_format", "")),
                },
            }
        )
    return documents, evidence_by_label


def _cohere_document_id(source: dict[str, Any], index: int) -> str:
    return str(source.get("chunk_id") or f"{source.get('doc_id', 'source')}_{index}")


def _message_text(response: Any) -> str:
    message = getattr(response, "message", None)
    content = getattr(message, "content", None) if message else None
    if isinstance(content, list):
        return "".join(str(getattr(part, "text", "") or "") for part in content).strip()
    return str(content or "").strip()


def _response_citations(response: Any) -> list[Any]:
    message = getattr(response, "message", None)
    citations = getattr(message, "citations", None) if message else None
    if not citations:
        return []
    return list(citations) if isinstance(citations, list) else [citations]


def _normalize_citations(
    raw_citations: list[Any],
    evidence_by_label: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for citation in raw_citations:
        sources: list[dict[str, Any]] = []
        for source in getattr(citation, "sources", []) or []:
            source_id = str(getattr(source, "id", "") or "")
            evidence = evidence_by_label.get(source_id, {})
            sources.append(
                {
                    "source_id": source_id,
                    "label": evidence.get("citation_id", source_id),
                    "doc_id": evidence.get("doc_id", ""),
                    "title": evidence.get("title", ""),
                    "page": evidence.get("page", ""),
                    "status": evidence.get("status", ""),
                    "version": evidence.get("version", ""),
                    "access_level": evidence.get("access_level", ""),
                    "language": evidence.get("language", ""),
                    "chunk_id": evidence.get("chunk_id", ""),
                    "canonical_url": evidence.get("canonical_url", ""),
                    "source_url": evidence.get("source_url", ""),
                    "source_pdf_url": evidence.get("source_pdf_url", ""),
                    "source_docx_url": evidence.get("source_docx_url", ""),
                    "retrieved_date": evidence.get("retrieved_date", ""),
                    "source_organization": evidence.get("source_organization", ""),
                    "provenance_note": evidence.get("provenance_note", ""),
                    "vector_score": evidence.get("vector_score"),
                    "rerank_score": evidence.get("rerank_score"),
                }
            )
        normalized.append(
            {
                "type": "cohere_native",
                "start": getattr(citation, "start", None),
                "end": getattr(citation, "end", None),
                "text": str(getattr(citation, "text", "") or ""),
                "sources": sources,
            }
        )
    return normalized


def _insert_inline_markers(answer: str, citations: list[dict[str, Any]]) -> str:
    cited_answer = answer
    for citation in sorted(citations, key=lambda item: int(item.get("end") or 0), reverse=True):
        labels = sorted({str(source.get("label", "")) for source in citation.get("sources", []) if source.get("label")})
        end = citation.get("end")
        if not labels or not isinstance(end, int) or end < 0 or end > len(cited_answer):
            continue
        marker = " [" + ", ".join(labels) + "]"
        if cited_answer[max(0, end - len(marker) - 2) : end + len(marker) + 2].find(marker) >= 0:
            continue
        cited_answer = cited_answer[:end] + marker + cited_answer[end:]
    return cited_answer


def _validate_native_citations(
    citations: list[dict[str, Any]],
    evidence_by_label: dict[str, dict[str, Any]],
    *,
    answer: str = "",
    query: str = "",
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not citations:
        errors.append("cohere_returned_no_native_citations")
    allowed_labels = set(evidence_by_label)
    cited_source_ids: set[str] = set()
    cited_doc_ids: set[str] = set()
    for citation in citations:
        source_ids = {str(source.get("source_id", "")) for source in citation.get("sources", [])}
        cited_source_ids.update(source_id for source_id in source_ids if source_id)
        cited_doc_ids.update(
            str(source.get("doc_id", ""))
            for source in citation.get("sources", [])
            if source.get("doc_id")
        )
        unknown = sorted(source_ids - allowed_labels)
        if unknown:
            errors.append(f"citation_references_unknown_source:{','.join(unknown)}")
        if not source_ids:
            errors.append("citation_has_no_sources")
    coverage = _citation_coverage(answer, citations) if answer else {}
    if coverage.get("uncited_claim_count", 0):
        message = f"uncited_claims:{coverage['uncited_claim_count']}"
        claim_count = int(coverage.get("claim_count", 0) or 0)
        covered_claim_count = int(coverage.get("covered_claim_count", 0) or 0)
        coverage_ratio = covered_claim_count / claim_count if claim_count else 0.0
        if citations and coverage_ratio >= 0.9:
            warnings.append(message)
        else:
            errors.append(message)
    available_doc_ids = {
        str(evidence.get("doc_id", ""))
        for evidence in evidence_by_label.values()
        if evidence.get("doc_id")
    }
    if _expects_multi_document_citations(query) and len(available_doc_ids) > 1 and len(cited_doc_ids) < 2:
        warnings.append("citations_reference_too_few_documents")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "citation_count": len(citations),
        "cited_source_ids": sorted(cited_source_ids),
        "cited_doc_ids": sorted(cited_doc_ids),
        "coverage": coverage,
    }


def _citation_coverage(answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    claim_spans = _claim_spans(answer)
    covered: list[dict[str, Any]] = []
    uncited: list[dict[str, Any]] = []
    citation_spans = [
        (citation.get("start"), citation.get("end"))
        for citation in citations
        if isinstance(citation.get("start"), int) and isinstance(citation.get("end"), int)
    ]
    for claim in claim_spans:
        start = int(claim["start"])
        end = int(claim["end"])
        is_covered = any(
            (citation_end > start and citation_start < end)
            or (end <= citation_start <= end + 8)
            for citation_start, citation_end in citation_spans
        )
        if is_covered:
            covered.append(claim)
        else:
            uncited.append(claim)
    return {
        "claim_count": len(claim_spans),
        "covered_claim_count": len(covered),
        "uncited_claim_count": len(uncited),
        "uncited_claims": [claim["text"] for claim in uncited[:5]],
    }


def _claim_spans(answer: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for line in re.finditer(r"[^\n]+", answer):
        line_text = line.group(0)
        for match in re.finditer(r".+?(?:(?<!\d)[.!?](?!\d)|$)", line_text):
            raw = match.group(0)
            left_trimmed = raw.lstrip(" \t\r\n-*•0123456789.)")
            text = left_trimmed.strip()
            if not _is_claim_like(text):
                continue
            start = line.start() + match.start() + (len(raw) - len(left_trimmed))
            end = start + len(text)
            spans.append({"start": start, "end": end, "text": text})
    return spans


def _is_claim_like(text: str) -> bool:
    clean = text.strip().lower()
    if len(re.sub(r"[^a-z0-9]", "", clean)) < 12:
        return False
    refusal_starts = (
        "i do not have enough",
        "there is not enough",
        "evidence is insufficient",
        "insufficient evidence",
        "no authorized evidence",
    )
    return not clean.startswith(refusal_starts)


def _expects_multi_document_citations(query: str) -> bool:
    text = query.lower()
    indicators = {
        "compare",
        "comparison",
        "overlap",
        "both",
        "summarize both",
        "planning brief",
        "planning update",
        "source pages",
    }
    return any(indicator in text for indicator in indicators)


def _citation_repair_feedback(validation: dict[str, Any]) -> str:
    errors = ", ".join(validation.get("errors", []) or ["none"])
    warnings = ", ".join(validation.get("warnings", []) or [])
    coverage = validation.get("coverage", {}) or {}
    uncited = "; ".join(str(claim) for claim in coverage.get("uncited_claims", [])[:3])
    parts = [
        "Rewrite the answer so every factual sentence is cited.",
        "Use one substantial paragraph or two compact paragraphs.",
        "Do not include any uncited setup, caveat, or conclusion sentence.",
        f"Previous citation validation errors: {errors}.",
    ]
    if warnings:
        parts.append(f"Previous citation validation warnings: {warnings}.")
    if uncited:
        parts.append(f"Uncited text to fix or remove: {uncited}.")
    return " ".join(parts)


def _validation_score(validation: dict[str, Any]) -> tuple[int, int, int, int]:
    coverage = validation.get("coverage", {}) or {}
    return (
        1 if validation.get("passed") else 0,
        -len(validation.get("warnings", []) or []),
        -len(validation.get("errors", []) or []),
        int(coverage.get("covered_claim_count", 0) or 0),
    )


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        key = str(source.get("chunk_id") or source.get("doc_id") or len(deduped))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped[:MAX_GROUNDING_SOURCES]
