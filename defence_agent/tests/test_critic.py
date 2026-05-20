from __future__ import annotations

import asyncio
import json

from defence_agent import critic


def _audit() -> dict[str, object]:
    return {
        "retrieval": {
            "sources_sent_to_answer": [
                {
                    "chunk_id": "DOC1_page_001",
                    "doc_id": "DOC1",
                    "title": "Authorized Source",
                    "page": 1,
                    "access_level": "unclassified",
                }
            ],
            "excluded_sources": [],
        },
        "generation": {
            "document_count": 1,
            "cohere_document_ids": ["DOC1_page_001"],
            "citation_resolution": {"passed": True, "errors": [], "warnings": []},
        },
        "citations": _citations(),
    }


def _citations() -> list[dict[str, object]]:
    return [
        {
            "text": "Alpha is supported",
            "sources": [{"source_id": "DOC1_page_001", "doc_id": "DOC1", "chunk_id": "DOC1_page_001"}],
        },
        {
            "text": "Beta is supported",
            "sources": [{"source_id": "DOC1_page_001", "doc_id": "DOC1", "chunk_id": "DOC1_page_001"}],
        },
    ]


def _sources() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "DOC1_page_001",
            "doc_id": "DOC1",
            "title": "Authorized Source",
            "page": 1,
            "access_level": "unclassified",
            "text": "Alpha is supported by the authorized source. Beta is not discussed.",
        }
    ]


def test_critic_agent_uses_adk_llm_agent_with_prompt() -> None:
    assert critic.critic_agent.name == "defence_agent_reviewer"
    assert "You are the Defence Agent Reviewer Agent" in critic.REVIEWER_INSTRUCTION
    assert "Return JSON only" in critic.REVIEWER_INSTRUCTION


def test_reviewer_reviews_longer_answers_by_default() -> None:
    assert critic.MAX_REVIEW_CITATIONS >= 24


def test_critic_scores_verified_citations_from_adk_json(monkeypatch) -> None:
    async def fake_run(message_text: str) -> str:
        payload = json.loads(message_text)
        assert payload["citations"][0]["evidence"][0]["text"].startswith("Alpha is supported")
        return json.dumps(
            {
                "summary": "One citation is supported and one is not.",
                "citation_reviews": [
                    {
                        "citation_index": 1,
                        "answer_span": "Alpha is supported",
                        "source_ids": ["DOC1_page_001"],
                        "verdict": "verified",
                        "reason": "The evidence states alpha is supported.",
                    },
                    {
                        "citation_index": 2,
                        "answer_span": "Beta is supported",
                        "source_ids": ["DOC1_page_001"],
                        "verdict": "unverified",
                        "reason": "The evidence does not discuss beta.",
                    },
                ],
                "overall_reason": "Citation support is below threshold.",
            }
        )

    monkeypatch.setattr(critic, "_run_critic_agent", fake_run)

    report = asyncio.run(
        critic.review_answer(
            query="What does DOC1 say?",
            answer="Alpha is supported. Beta is supported.",
            citations=_citations(),
            answer_audit=_audit(),
            sources_sent_to_answer=_sources(),
            threshold=0.8,
        )
    )

    assert report["status"] == critic.NEEDS_REVISION
    assert report["credibility_score"] == 0.5
    assert report["trust_score"] == 0.5
    assert report["trust_label"] == "low_trust_revise"
    assert report["citation_trust_counts"] == {"green": 1, "yellow": 0, "red": 1, "unreviewed": 0}
    assert report["citation_reviews"][1]["trust_color"] == "red"
    assert report["verified_citation_count"] == 1
    assert report["unverified_citation_count"] == 1
    assert report["release_gate"] == "revise"
    assert report["requires_human_decision"] is False
    assert "Beta" in report["generator_feedback"] or "citation 2" in report["generator_feedback"]


def test_critic_approves_when_score_meets_threshold(monkeypatch) -> None:
    async def fake_run(_: str) -> str:
        return json.dumps(
            {
                "summary": "Both citations are supported.",
                "citation_reviews": [
                    {"citation_index": 1, "verdict": "verified", "reason": "supported"},
                    {"citation_index": 2, "verdict": "verified", "reason": "supported"},
                ],
                "overall_reason": "All reviewed citations were verified.",
            }
        )

    monkeypatch.setattr(critic, "_run_critic_agent", fake_run)

    report = asyncio.run(
        critic.review_answer(
            query="What does DOC1 say?",
            answer="Alpha is supported. Beta is supported.",
            citations=_citations(),
            answer_audit=_audit(),
            sources_sent_to_answer=_sources(),
            threshold=0.8,
        )
    )

    assert report["status"] == critic.APPROVED
    assert report["credibility_score"] == 1.0
    assert report["trust_label"] == "high_trust"
    assert report["release_gate"] == "release"
    assert report["requires_human_decision"] is False


def test_critic_blocks_release_when_any_citation_is_unverified_even_above_threshold(monkeypatch) -> None:
    citations = [
        {
            "text": f"Claim {index}",
            "sources": [{"source_id": "DOC1_page_001", "doc_id": "DOC1", "chunk_id": "DOC1_page_001"}],
        }
        for index in range(1, 6)
    ]
    audit = _audit()
    audit["citations"] = citations

    async def fake_run(_: str) -> str:
        return json.dumps(
            {
                "summary": "Four citations are supported and one is not.",
                "citation_reviews": [
                    {"citation_index": 1, "verdict": "verified", "reason": "supported"},
                    {"citation_index": 2, "verdict": "verified", "reason": "supported"},
                    {"citation_index": 3, "verdict": "verified", "reason": "supported"},
                    {"citation_index": 4, "verdict": "verified", "reason": "supported"},
                    {"citation_index": 5, "verdict": "unverified", "reason": "unsupported"},
                ],
                "overall_reason": "One citation does not support the answer span.",
            }
        )

    monkeypatch.setattr(critic, "_run_critic_agent", fake_run)

    report = asyncio.run(
        critic.review_answer(
            query="What does DOC1 say?",
            answer="Claim 1. Claim 2. Claim 3. Claim 4. Claim 5.",
            citations=citations,
            answer_audit=audit,
            sources_sent_to_answer=_sources(),
            threshold=0.8,
        )
    )

    assert report["credibility_score"] == 0.8
    assert report["status"] == critic.NEEDS_REVISION
    assert report["release_gate"] == "revise"
    assert report["low_trust_citation_count"] == 1


def test_critic_falls_back_to_resolved_audit_citations(monkeypatch) -> None:
    class RawCitation:
        pass

    async def fake_run(message_text: str) -> str:
        payload = json.loads(message_text)
        assert len(payload["citations"]) == 2
        assert payload["citations"][0]["source_ids"] == ["DOC1_page_001"]
        return json.dumps(
            {
                "summary": "Resolved audit citations were reviewed.",
                "citation_reviews": [
                    {"citation_index": 1, "verdict": "verified", "reason": "supported"},
                    {"citation_index": 2, "verdict": "verified", "reason": "supported"},
                ],
                "overall_reason": "All reviewed citations were verified.",
            }
        )

    monkeypatch.setattr(critic, "_run_critic_agent", fake_run)

    report = asyncio.run(
        critic.review_answer(
            query="What does DOC1 say?",
            answer="Alpha is supported. Beta is supported.",
            citations=[RawCitation()],  # type: ignore[list-item]
            answer_audit=_audit(),
            sources_sent_to_answer=_sources(),
            threshold=0.8,
        )
    )

    assert report["status"] == critic.APPROVED
    assert report["total_citation_count"] == 2
    assert report["verified_citation_count"] == 2
    assert report["unreviewed_citation_count"] == 0


def test_critic_uses_retrieval_chunk_text_when_citation_targets_child_chunk(monkeypatch) -> None:
    audit = _audit()
    audit["citations"] = [
        {
            "text": "Needle is supported",
            "sources": [
                {
                    "source_id": "DOC1_page_001_chunk_002",
                    "doc_id": "DOC1",
                    "chunk_id": "DOC1_page_001_chunk_002",
                }
            ],
        }
    ]
    sources = [
        {
            "chunk_id": "DOC1_page_001",
            "parent_page_id": "DOC1_page_001",
            "retrieval_chunk_id": "DOC1_page_001_chunk_002",
            "retrieval_chunk_text": "Needle is supported in the child chunk.",
            "chunk_strategy": "windowed",
            "doc_id": "DOC1",
            "title": "Authorized Source",
            "page": 1,
            "access_level": "unclassified",
            "text": "Long parent page text.",
        }
    ]

    async def fake_run(message_text: str) -> str:
        payload = json.loads(message_text)
        evidence = payload["citations"][0]["evidence"][0]
        assert evidence["source_id"] == "DOC1_page_001_chunk_002"
        assert evidence["text_granularity"] == "retrieval_chunk"
        assert evidence["text"].startswith("Needle is supported")
        return json.dumps(
            {
                "summary": "The child-chunk citation is supported.",
                "citation_reviews": [{"citation_index": 1, "verdict": "verified", "reason": "supported"}],
                "overall_reason": "The cited child chunk supports the answer span.",
            }
        )

    monkeypatch.setattr(critic, "_run_critic_agent", fake_run)

    report = asyncio.run(
        critic.review_answer(
            query="What does DOC1 say?",
            answer="Needle is supported.",
            citations=[],
            answer_audit=audit,
            sources_sent_to_answer=sources,
            threshold=0.8,
        )
    )

    assert report["status"] == critic.APPROVED
    assert report["verified_citation_count"] == 1


def test_critic_preserves_explicit_generator_feedback(monkeypatch) -> None:
    async def fake_run(_: str) -> str:
        return json.dumps(
            {
                "summary": "The citation is too weak.",
                "citation_reviews": [
                    {"citation_index": 1, "verdict": "unverified", "reason": "missing the specific number"},
                    {"citation_index": 2, "verdict": "verified", "reason": "supported"},
                ],
                "generator_feedback": "Search for the exact threshold and remove unsupported numeric claims.",
                "suggested_search_queries": ["exact threshold release control"],
                "overall_reason": "One citation is unsupported.",
            }
        )

    monkeypatch.setattr(critic, "_run_critic_agent", fake_run)

    report = asyncio.run(
        critic.review_answer(
            query="What does DOC1 say?",
            answer="Alpha is supported. Beta is supported.",
            citations=_citations(),
            answer_audit=_audit(),
            sources_sent_to_answer=_sources(),
            threshold=0.8,
        )
    )

    assert report["status"] == critic.NEEDS_REVISION
    assert "exact threshold" in report["generator_feedback"]
    assert report["suggested_search_queries"] == ["exact threshold release control"]


def test_critic_approves_refusal_without_api_call(monkeypatch) -> None:
    async def fail_if_called(_: str) -> str:
        raise AssertionError("refusal review should not call the critic model")

    monkeypatch.setattr(critic, "_run_critic_agent", fail_if_called)
    audit = {
        "retrieval": {"sources_sent_to_answer": [], "excluded_sources": [{"doc_id": "DOC_SECRET"}]},
        "generation": {"document_count": 0, "cohere_document_ids": []},
    }

    report = asyncio.run(
        critic.review_answer(
            query="What is the secret?",
            answer="I do not have enough authorized evidence to answer.",
            citations=[],
            answer_audit=audit,
            sources_sent_to_answer=[],
            threshold=0.8,
        )
    )

    assert report["status"] == critic.APPROVED
    assert report["credibility_score"] == 1.0
    assert report["total_citation_count"] == 0


def test_critic_gates_refusal_that_releases_citations(monkeypatch) -> None:
    async def fail_if_called(_: str) -> str:
        raise AssertionError("released refusal should not call the critic model")

    monkeypatch.setattr(critic, "_run_critic_agent", fail_if_called)

    report = asyncio.run(
        critic.review_answer(
            query="What is unsupported?",
            answer="I do not have enough cited evidence from the authorized documents to answer this question.",
            citations=_citations(),
            answer_audit=_audit(),
            sources_sent_to_answer=_sources(),
            threshold=0.8,
        )
    )

    assert report["status"] == critic.NEEDS_HUMAN_REVIEW
    assert report["credibility_score"] == 0.0
    assert report["answer_citation_count"] == 2
    assert report["total_citation_count"] == 2
    assert report["release_gate"] == "human_continue_or_stop_required"
