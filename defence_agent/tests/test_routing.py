from __future__ import annotations

from defence_agent.routing import AGENTIC_RAG, AUTO, REVIEWED_AGENT, SIMPLE_RAG, choose_route


def test_auto_route_uses_reviewer_for_high_accuracy_cited_questions() -> None:
    route = choose_route(
        requested_mode=AUTO,
        query="Compare the policy sources and cite the strongest pages.",
        accuracy_priority=5,
        latency_priority=1,
    )

    assert route.selected_mode == REVIEWED_AGENT
    assert route.uses_adk_agent is True
    assert route.uses_reviewer is True
    assert route.expected_latency == "high"


def test_auto_route_can_choose_simple_rag_for_low_stakes_latency() -> None:
    route = choose_route(
        requested_mode=AUTO,
        query="What does the AI strategy say about training?",
        accuracy_priority=1,
        latency_priority=5,
    )

    assert route.selected_mode == SIMPLE_RAG
    assert route.uses_adk_agent is False
    assert route.uses_reviewer is False
    assert route.expected_cost == "low"


def test_explicit_agentic_route_skips_reviewer() -> None:
    route = choose_route(
        requested_mode=AGENTIC_RAG,
        query="Compare two documents.",
        accuracy_priority=5,
        latency_priority=1,
    )

    assert route.selected_mode == AGENTIC_RAG
    assert route.uses_adk_agent is True
    assert route.uses_reviewer is False
