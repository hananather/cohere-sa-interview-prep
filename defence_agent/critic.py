"""ADK reviewer agent for reviewed answers.

The research agent remains the existing ADK retrieval path plus Cohere grounded
generation. This module adds one bounded ADK LLM reviewer pass that checks
whether cited evidence supports the answer claims and returns a citation
credibility score.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from defence_agent.cohere_gateway import cohere_gateway


APPROVED = "approved"
NEEDS_REVISION = "needs_revision"
NEEDS_CLARIFICATION = "needs_clarification"
NEEDS_HUMAN_REVIEW = "needs_human_review"
DEFAULT_CREDIBILITY_THRESHOLD = float(
    os.getenv("DEFTECH_ADK_REVIEWER_THRESHOLD", os.getenv("DEFTECH_ADK_CRITIC_THRESHOLD", "0.8"))
)
MAX_EVIDENCE_CHARS = 1600
MAX_REVIEW_CITATIONS = int(
    os.getenv("DEFTECH_ADK_REVIEWER_MAX_CITATIONS", os.getenv("DEFTECH_ADK_CRITIC_MAX_CITATIONS", "24"))
)

REVIEWER_AGENT_ID = "defence_agent_reviewer"
REVIEWER_MODEL = os.getenv("DEFTECH_ADK_REVIEWER_MODEL", os.getenv("DEFTECH_ADK_CRITIC_MODEL", "command-a-reasoning-08-2025"))
REVIEWER_TIMEOUT_SECONDS = float(
    os.getenv("DEFTECH_ADK_REVIEWER_TIMEOUT_SECONDS", os.getenv("DEFTECH_ADK_CRITIC_TIMEOUT_SECONDS", "45"))
)

# Backward-compatible names for existing imports and audit readers.
CRITIC_AGENT_ID = REVIEWER_AGENT_ID
CRITIC_MODEL = REVIEWER_MODEL
CRITIC_TIMEOUT_SECONDS = REVIEWER_TIMEOUT_SECONDS

REVIEWER_INSTRUCTION = """
You are the Defence Agent Reviewer Agent.

Review the Research Agent's final answer against only the supplied cited evidence.
Do not answer the user's original question. Do not add new facts.

Your job:
- For each citation, decide whether the cited evidence supports the cited answer span.
- Review every citation object supplied in the input.
- Mark a citation "verified" only when the evidence directly supports the answer span.
- Mark a citation "unverified" when the evidence does not support the span or the support is too weak.
- Mark a citation "unclear" when the supplied evidence is insufficient to decide.
- Treat "unclear" as not verified.
- If the answer is a refusal because no authorized evidence was sent, approve the refusal only if no citation or source text is used.
- Do not treat excluded source metadata as evidence. Excluded source text is intentionally unavailable.

Return JSON only with this shape:
{
  "summary": "one sentence",
  "citation_reviews": [
    {
      "citation_index": 1,
      "verdict": "verified|unverified|unclear",
      "reason": "short reason"
    }
  ],
  "overall_reason": "short reason"
}
""".strip()

CRITIC_INSTRUCTION = REVIEWER_INSTRUCTION


class CohereChatReviewerAgent(BaseAgent):
    """ADK agent that uses Cohere Chat V2 as its model transport."""

    model: str
    instruction: str

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        response = cohere_gateway.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": _content_text(ctx.user_content)},
            ],
            temperature=0.0,
        )
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=_cohere_response_text(response))],
            ),
        )


def _cohere_model_id(model: str) -> str:
    return model.removeprefix("cohere/")


critic_agent = CohereChatReviewerAgent(
    name=CRITIC_AGENT_ID,
    model=_cohere_model_id(CRITIC_MODEL),
    instruction=REVIEWER_INSTRUCTION,
)


async def review_turn_result(
    result: Any,
    *,
    threshold: float = DEFAULT_CREDIBILITY_THRESHOLD,
) -> dict[str, Any]:
    """Run the ADK reviewer over an ``AgentTurnResult``-like object."""

    return await review_answer(
        query=str(getattr(result, "answer_audit", {}).get("query", "") or ""),
        answer=str(getattr(result, "answer", "") or ""),
        citations=list(getattr(result, "citations", []) or []),
        answer_audit=dict(getattr(result, "answer_audit", {}) or {}),
        sources_sent_to_answer=[],
        threshold=threshold,
    )


async def review_answer(
    *,
    query: str,
    answer: str,
    citations: list[dict[str, Any]],
    answer_audit: dict[str, Any],
    sources_sent_to_answer: list[dict[str, Any]],
    threshold: float = DEFAULT_CREDIBILITY_THRESHOLD,
) -> dict[str, Any]:
    """Run an ADK LLM reviewer and return a scored review report."""

    citations_for_review = _best_review_citations(citations, answer_audit)
    review_input = _review_input(
        query=query,
        answer=answer,
        citations=citations_for_review,
        answer_audit=answer_audit,
        sources_sent_to_answer=sources_sent_to_answer,
        threshold=threshold,
    )
    if review_input["answer_type"] == "refusal":
        return _refusal_report(review_input, threshold=threshold)

    critic_text = ""
    try:
        critic_text = await _run_critic_agent(json.dumps(review_input, ensure_ascii=False, indent=2))
        critic_payload = _parse_json_object(critic_text)
        return _score_report(review_input, critic_payload, threshold=threshold)
    except Exception as exc:
        return _critic_error_report(review_input, exc, threshold=threshold, raw_text=critic_text)


async def _run_critic_agent(message_text: str) -> str:
    session_service = InMemorySessionService()
    session_id = f"reviewer_{uuid4().hex}"
    user_id = "reviewed_answer_mode"
    await _maybe_await(
        session_service.create_session(
            app_name=CRITIC_AGENT_ID,
            user_id=user_id,
            session_id=session_id,
            state={},
        )
    )
    runner = Runner(app_name=CRITIC_AGENT_ID, agent=critic_agent, session_service=session_service)
    message = types.Content(role="user", parts=[types.Part.from_text(text=message_text)])
    final_text = ""
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
        if event.is_final_response():
            final_text = _event_text(event)
    return final_text


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _event_text(event: Any) -> str:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(str(getattr(part, "text", "") or "") for part in parts).strip()


def _content_text(content: Any) -> str:
    parts = getattr(content, "parts", None) or []
    return "".join(str(getattr(part, "text", "") or "") for part in parts).strip()


def _cohere_response_text(response: Any) -> str:
    message = getattr(response, "message", None)
    content = getattr(message, "content", None) if message else None
    if isinstance(content, list):
        return "".join(
            str(getattr(part, "text", "") or "")
            for part in content
            if str(getattr(part, "type", "") or "") == "text"
        ).strip()
    return str(content or "").strip()


def _review_input(
    *,
    query: str,
    answer: str,
    citations: list[dict[str, Any]],
    answer_audit: dict[str, Any],
    sources_sent_to_answer: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    retrieval = _dict(answer_audit.get("retrieval"))
    generation = _dict(answer_audit.get("generation"))
    all_citation_items = _citation_items(citations)
    answer_type = _answer_type(answer, all_citation_items, generation)
    citation_items = [] if answer_type == "refusal" else _bounded_citation_items(all_citation_items)
    evidence_lookup = _evidence_lookup(sources_sent_to_answer)
    return {
        "schema_version": "reviewer_input.v1",
        "threshold": threshold,
        "answer_citation_count": len(all_citation_items),
        "reviewed_citation_limit": MAX_REVIEW_CITATIONS,
        "query": query,
        "answer": answer,
        "answer_type": answer_type,
        "citation_validation": generation.get("citation_resolution", {}),
        "citations": [
            {
                **item,
                "evidence": [
                    evidence_lookup.get(source_id, {"source_id": source_id, "text": ""})
                    for source_id in item["source_ids"]
                ],
            }
            for item in citation_items
        ],
        "excluded_sources": retrieval.get("excluded_sources", []),
        "limits": {
            "excluded_source_text_available": False,
            "truth_verification": "not_claimed",
            "human_approval": "not_claimed",
        },
    }


def _citation_items(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, citation in enumerate(citations or [], start=1):
        if not isinstance(citation, dict):
            continue
        source_ids = []
        for source in citation.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            source_id = _source_identity(source)
            if source_id:
                source_ids.append(source_id)
        items.append(
            {
                "citation_index": index,
                "answer_span": str(citation.get("text", "") or ""),
                "source_ids": sorted(set(source_ids)),
            }
        )
    return items


def _best_review_citations(citations: list[dict[str, Any]], answer_audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer the citation list with resolved source IDs.

    Raw SDK citation objects are not always plain dictionaries. The answer audit
    contains the resolved source metadata used by the UI and eval harness, so
    the reviewer should use it when it is more reviewable.
    """

    candidates = [
        [citation for citation in citations or [] if isinstance(citation, dict)],
        [citation for citation in answer_audit.get("citations", []) or [] if isinstance(citation, dict)],
    ]
    return max(candidates, key=_reviewability_score, default=[])


def _reviewability_score(citations: list[dict[str, Any]]) -> tuple[int, int, int]:
    items = _citation_items(citations)
    source_count = sum(len(item.get("source_ids", []) or []) for item in items)
    sourced_citation_count = sum(1 for item in items if item.get("source_ids"))
    return source_count, sourced_citation_count, len(items)


def _source_identity(source: dict[str, Any]) -> str:
    for field in ("source_id", "chunk_id", "cohere_document_id", "citation_id"):
        value = str(source.get(field, "") or "").strip()
        if value:
            return value
    doc_id = str(source.get("doc_id", "") or "").strip()
    page = str(source.get("page", "") or "").strip()
    if doc_id and page:
        try:
            page = f"{int(page):03d}"
        except ValueError:
            pass
        return f"{doc_id}_page_{page}"
    return ""


def _bounded_citation_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(items) <= MAX_REVIEW_CITATIONS:
        return items
    if MAX_REVIEW_CITATIONS <= 1:
        return [items[0]]
    selected_indexes = {
        round(index * (len(items) - 1) / (MAX_REVIEW_CITATIONS - 1))
        for index in range(MAX_REVIEW_CITATIONS)
    }
    return [items[index] for index in sorted(selected_indexes)]


def _evidence_lookup(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for source_id in _evidence_aliases(source):
            lookup[source_id] = _evidence_entry(source, source_id)
    return lookup


def _evidence_aliases(source: dict[str, Any]) -> set[str]:
    aliases = {
        str(source.get("chunk_id", "") or ""),
        str(source.get("source_id", "") or ""),
        str(source.get("retrieval_chunk_id", "") or ""),
        str(source.get("parent_page_id", "") or ""),
        _source_identity(source),
    }
    return {alias for alias in aliases if alias}


def _evidence_entry(source: dict[str, Any], source_id: str) -> dict[str, Any]:
    retrieval_chunk_id = str(source.get("retrieval_chunk_id", "") or "")
    retrieval_chunk_text = str(source.get("retrieval_chunk_text", "") or "")
    use_retrieval_chunk = bool(retrieval_chunk_text and retrieval_chunk_id and source_id == retrieval_chunk_id)
    text = retrieval_chunk_text if use_retrieval_chunk else str(source.get("text", "") or "")
    return {
        "source_id": source_id,
        "doc_id": str(source.get("doc_id", "") or ""),
        "title": str(source.get("title", "") or ""),
        "page": source.get("page", ""),
        "access_level": str(source.get("access_level", "") or ""),
        "chunk_strategy": str(source.get("chunk_strategy", "") or ""),
        "chunk_id": str(source.get("chunk_id", "") or ""),
        "parent_page_id": str(source.get("parent_page_id", "") or ""),
        "retrieval_chunk_id": retrieval_chunk_id,
        "text_granularity": "retrieval_chunk" if use_retrieval_chunk else "parent_page",
        "text": text[:MAX_EVIDENCE_CHARS],
    }


def _answer_type(answer: str, citation_items: list[dict[str, Any]], generation: dict[str, Any]) -> str:
    if str(generation.get("citation_mode", "")) == "answer_audit_lookup":
        return "audit_lookup"
    if _is_refusal_answer(answer):
        return "refusal"
    return "cited_answer"


def _is_refusal_answer(answer: str) -> bool:
    normalized = " ".join(answer.lower().split())
    markers = (
        "i do not have enough",
        "there is not enough",
        "evidence is insufficient",
        "insufficient evidence",
        "no authorized evidence",
    )
    return any(marker in normalized for marker in markers)


def _refusal_report(review_input: dict[str, Any], *, threshold: float) -> dict[str, Any]:
    answer_citation_count = int(review_input.get("answer_citation_count", 0) or 0)
    no_citations = answer_citation_count == 0
    score = 1.0 if no_citations else 0.0
    status = APPROVED if score >= threshold else NEEDS_HUMAN_REVIEW
    summary = (
        "Refusal reviewed without using excluded source text."
        if no_citations
        else "Refusal included released citations and requires human review."
    )
    overall_reason = (
        "The answer is a refusal and no citations were released."
        if no_citations
        else "A refusal should not release citation/source text as supporting evidence."
    )
    return _report(
        status=status,
        credibility_score=score,
        verified_count=0,
        unverified_count=0 if no_citations else answer_citation_count,
        total_citations=0 if no_citations else answer_citation_count,
        answer_citation_count=answer_citation_count,
        threshold=threshold,
        citation_reviews=[],
        summary=summary,
        overall_reason=overall_reason,
        raw_critic_response={},
    )


def _score_report(
    review_input: dict[str, Any],
    critic_payload: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    expected_citations = [item for item in review_input.get("citations", []) or [] if isinstance(item, dict)]
    citation_reviews = _normalized_reviews(
        critic_payload.get("citation_reviews", []),
        expected_citations=expected_citations,
    )
    total = len(expected_citations)
    verified = sum(1 for item in citation_reviews if item["verdict"] == "verified")
    unverified = max(total - verified, 0)
    score = round(verified / total, 4) if total else 0.0
    # A mostly good average is not enough for release in this workflow. One
    # unsupported citation should force revision because the answer is only as
    # trusted as the evidence attached to each claim.
    if total and unverified == 0 and score >= threshold:
        status = APPROVED
    elif total:
        status = NEEDS_REVISION
    else:
        status = NEEDS_CLARIFICATION
    summary = str(critic_payload.get("summary", "") or "")
    output_validation = _critic_output_validation(
        summary=summary,
        raw_review_count=len([item for item in critic_payload.get("citation_reviews", []) or [] if isinstance(item, dict)]),
        expected_review_count=total,
        unverified_count=unverified,
    )
    if not output_validation["passed"]:
        status = NEEDS_HUMAN_REVIEW
        if "summary_conflicts_with_citation_reviews" in output_validation["warnings"]:
            summary = f"{verified} of {total} citations were verified by the reviewer."
    generator_feedback = _generator_feedback(
        status=status,
        critic_payload=critic_payload,
        citation_reviews=citation_reviews,
    )
    return _report(
        status=status,
        credibility_score=score,
        verified_count=verified,
        unverified_count=unverified,
        total_citations=total,
        answer_citation_count=int(review_input.get("answer_citation_count", total) or total),
        threshold=threshold,
        citation_reviews=citation_reviews,
        summary=summary,
        overall_reason=str(critic_payload.get("overall_reason", "") or ""),
        raw_critic_response=critic_payload,
        critic_output_validation=output_validation,
        generator_feedback=generator_feedback,
        suggested_search_queries=_suggested_search_queries(critic_payload),
    )


def _normalized_reviews(raw_reviews: Any, *, expected_citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reviews = [item for item in raw_reviews or [] if isinstance(item, dict)]
    normalized: list[dict[str, Any]] = []
    for expected in expected_citations:
        index = int(expected.get("citation_index", 0) or 0)
        item = next((entry for entry in reviews if int(entry.get("citation_index", 0) or 0) == index), {})
        verdict = str(item.get("verdict", "unclear") or "unclear").lower()
        if verdict not in {"verified", "unverified", "unclear"}:
            verdict = "unclear"
        normalized.append(
            {
                "citation_index": index,
                "answer_span": str(expected.get("answer_span", "") or item.get("answer_span", "") or ""),
                "source_ids": [
                    str(source_id)
                    for source_id in (expected.get("source_ids", []) or item.get("source_ids", []) or [])
                ],
                "verdict": verdict,
                "trust_color": _trust_color(verdict),
                "reason": str(item.get("reason", "") or ""),
            }
        )
    return normalized


def _critic_error_report(
    review_input: dict[str, Any],
    exc: Exception,
    *,
    threshold: float,
    raw_text: str = "",
) -> dict[str, Any]:
    return _report(
        status=NEEDS_HUMAN_REVIEW,
        credibility_score=0.0,
        verified_count=0,
        unverified_count=len(review_input.get("citations", [])),
        total_citations=len(review_input.get("citations", [])),
        answer_citation_count=int(review_input.get("answer_citation_count", len(review_input.get("citations", []))) or 0),
        threshold=threshold,
        citation_reviews=[],
        summary="Reviewer review failed.",
        overall_reason=f"{type(exc).__name__}: {exc}",
        raw_critic_response={"raw_text": raw_text[:4000]} if raw_text else {},
        critic_output_validation={"passed": False, "warnings": ["critic_call_failed"]},
        generator_feedback="",
        suggested_search_queries=[],
    )


def _report(
    *,
    status: str,
    credibility_score: float,
    verified_count: int,
    unverified_count: int,
    total_citations: int,
    answer_citation_count: int | None = None,
    threshold: float,
    citation_reviews: list[dict[str, Any]],
    summary: str,
    overall_reason: str,
    raw_critic_response: dict[str, Any],
    critic_output_validation: dict[str, Any] | None = None,
    generator_feedback: str = "",
    suggested_search_queries: list[str] | None = None,
) -> dict[str, Any]:
    requires_human_decision = status in {NEEDS_CLARIFICATION, NEEDS_HUMAN_REVIEW}
    trust_counts = _citation_trust_counts(
        citation_reviews,
        answer_citation_count=answer_citation_count if answer_citation_count is not None else total_citations,
        fallback_unverified_count=unverified_count,
    )
    return {
        "schema_version": "reviewer_report.v1",
        "reviewer": CRITIC_AGENT_ID,
        "model": CRITIC_MODEL,
        "status": status,
        "trust_score": credibility_score,
        "trust_label": _trust_label(status),
        "credibility_score": credibility_score,
        "threshold": threshold,
        "verified_citation_count": verified_count,
        "unverified_citation_count": unverified_count,
        "total_citation_count": total_citations,
        "answer_citation_count": answer_citation_count if answer_citation_count is not None else total_citations,
        "unreviewed_citation_count": max((answer_citation_count or total_citations) - total_citations, 0),
        "low_trust_citation_count": trust_counts["yellow"] + trust_counts["red"],
        "citation_trust_counts": trust_counts,
        "citation_reviews": citation_reviews,
        "release_gate": _release_gate(status),
        "requires_human_decision": requires_human_decision,
        "human_prompt": (
            "The previous output did not meet the citation credibility threshold. "
            "Rerun with the reviewer feedback or remove unsupported claims before release."
            if requires_human_decision
            else ""
        ),
        "generator_feedback": generator_feedback,
        "suggested_search_queries": suggested_search_queries or [],
        "summary": summary,
        "overall_reason": overall_reason,
        "critic_output_validation": critic_output_validation or {"passed": True, "warnings": []},
        "raw_critic_response": raw_critic_response,
        "limits": {
            "truth_verification": "not_claimed",
            "human_approval": "not_claimed",
            "dynamic_multi_agent_fanout": "not_used",
        },
    }


def _trust_color(verdict: str) -> str:
    if verdict == "verified":
        return "green"
    if verdict == "unclear":
        return "yellow"
    return "red"


def _trust_label(status: str) -> str:
    if status == APPROVED:
        return "high_trust"
    if status == NEEDS_REVISION:
        return "low_trust_revise"
    if status == NEEDS_CLARIFICATION:
        return "insufficient_citations"
    return "low_trust_human_review"


def _citation_trust_counts(
    citation_reviews: list[dict[str, Any]],
    *,
    answer_citation_count: int,
    fallback_unverified_count: int,
) -> dict[str, int]:
    counts = {"green": 0, "yellow": 0, "red": 0, "unreviewed": 0}
    for review in citation_reviews:
        color = str(review.get("trust_color") or _trust_color(str(review.get("verdict", "") or "")))
        if color in {"green", "yellow", "red"}:
            counts[color] += 1
    reviewed = counts["green"] + counts["yellow"] + counts["red"]
    if reviewed == 0 and fallback_unverified_count:
        counts["red"] = fallback_unverified_count
        reviewed = fallback_unverified_count
    counts["unreviewed"] = max(int(answer_citation_count or 0) - reviewed, 0)
    return counts


def _release_gate(status: str) -> str:
    if status == APPROVED:
        return "release"
    if status == NEEDS_REVISION:
        return "revise"
    if status == NEEDS_CLARIFICATION:
        return "clarification_required"
    return "human_continue_or_stop_required"


def _generator_feedback(
    *,
    status: str,
    critic_payload: dict[str, Any],
    citation_reviews: list[dict[str, Any]],
) -> str:
    raw_feedback = str(critic_payload.get("generator_feedback", "") or "").strip()
    if status != NEEDS_REVISION:
        return raw_feedback[:1200]
    unsupported = [
        review
        for review in citation_reviews
        if review.get("verdict") in {"unverified", "unclear"}
    ][:4]
    if raw_feedback:
        prefix = raw_feedback
    else:
        prefix = (
            "Revise the answer so every factual claim is directly supported by authorized cited evidence. "
            "Search again if the current evidence does not support the claim."
        )
    if not unsupported:
        return prefix[:1200]
    details = "; ".join(
        f"citation {item.get('citation_index')}: {str(item.get('reason', '') or 'support was weak')[:180]}"
        for item in unsupported
    )
    return f"{prefix} Weak or unsupported citations: {details}"[:1200]


def _suggested_search_queries(critic_payload: dict[str, Any]) -> list[str]:
    raw = critic_payload.get("suggested_search_queries", [])
    if not isinstance(raw, list):
        return []
    queries: list[str] = []
    for item in raw:
        query = " ".join(str(item or "").split())
        if query and query not in queries:
            queries.append(query[:240])
        if len(queries) >= 3:
            break
    return queries


def _critic_output_validation(
    *,
    summary: str,
    raw_review_count: int,
    expected_review_count: int,
    unverified_count: int,
) -> dict[str, Any]:
    warnings: list[str] = []
    if raw_review_count != expected_review_count:
        warnings.append("critic_review_count_mismatch")
    lowered = summary.lower()
    overclaims_full_support = any(
        phrase in lowered
        for phrase in (
            "fully verified",
            "all citations are verified",
            "all reviewed citations were verified",
            "all claims",
        )
    )
    if unverified_count and overclaims_full_support:
        warnings.append("summary_conflicts_with_citation_reviews")
    return {"passed": not warnings, "warnings": warnings}


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Reviewer returned JSON that was not an object.")
    return parsed


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
