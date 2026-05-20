"""Small routing policy for demo autonomy levels.

The router is intentionally deterministic. It makes the latency, cost, and
accuracy trade-off visible without adding another model call before every demo
question.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


AUTO = "auto"
SIMPLE_RAG = "simple_rag"
AGENTIC_RAG = "agentic_rag"
REVIEWED_AGENT = "reviewed_agent"

SUPPORTED_RUN_MODES = {AUTO, SIMPLE_RAG, AGENTIC_RAG, REVIEWED_AGENT}

RUN_MODE_LABELS = {
    AUTO: "Auto route",
    SIMPLE_RAG: "RAG",
    AGENTIC_RAG: "Agentic RAG",
    REVIEWED_AGENT: "Multi-agent",
}


@dataclass(frozen=True)
class RouteDecision:
    requested_mode: str
    selected_mode: str
    label: str
    reason: str
    accuracy_priority: int
    latency_priority: int
    expected_latency: str
    expected_cost: str
    uses_retrieval: bool
    uses_adk_agent: bool
    uses_reviewer: bool

    def as_audit(self) -> dict[str, object]:
        return asdict(self)


def choose_route(
    *,
    requested_mode: str,
    query: str,
    accuracy_priority: int = 4,
    latency_priority: int = 2,
) -> RouteDecision:
    """Choose the demo route for a query.

    ``accuracy_priority`` and ``latency_priority`` are simple 1..5 UI inputs.
    Auto mode favors multi-agent citation review for high-stakes or source-sensitive
    questions and favors RAG only when the user explicitly optimizes for
    latency over accuracy.
    """

    requested = _normalize_mode(requested_mode)
    accuracy = _bounded_priority(accuracy_priority)
    latency = _bounded_priority(latency_priority)
    selected = requested if requested != AUTO else _auto_route(query, accuracy=accuracy, latency=latency)
    return _decision(
        requested_mode=requested,
        selected_mode=selected,
        reason=_route_reason(requested, selected, query=query, accuracy=accuracy, latency=latency),
        accuracy_priority=accuracy,
        latency_priority=latency,
    )


def reviewer_enabled(mode: str) -> bool:
    return _normalize_mode(mode) == REVIEWED_AGENT


def _auto_route(query: str, *, accuracy: int, latency: int) -> str:
    text = query.lower()
    high_stakes_terms = {
        "restricted",
        "secret",
        "threshold",
        "exact",
        "cite",
        "citation",
        "source",
        "page",
        "approved",
        "compare",
        "policy",
        "doctrine",
        "manual",
        "evidence",
    }
    multi_step_terms = {"compare", "both", "versus", "across", "synthesize", "brief", "planning"}
    high_stakes = any(term in text for term in high_stakes_terms)
    multi_step = any(term in text for term in multi_step_terms)

    if accuracy >= 4 or high_stakes:
        return REVIEWED_AGENT
    if latency >= 4 and accuracy <= 2 and not multi_step:
        return SIMPLE_RAG
    if multi_step or len(query.split()) > 24:
        return AGENTIC_RAG
    return SIMPLE_RAG


def _decision(
    *,
    requested_mode: str,
    selected_mode: str,
    reason: str,
    accuracy_priority: int,
    latency_priority: int,
) -> RouteDecision:
    metadata = {
        SIMPLE_RAG: {
            "expected_latency": "low",
            "expected_cost": "low",
            "uses_adk_agent": False,
            "uses_reviewer": False,
        },
        AGENTIC_RAG: {
            "expected_latency": "medium",
            "expected_cost": "medium",
            "uses_adk_agent": True,
            "uses_reviewer": False,
        },
        REVIEWED_AGENT: {
            "expected_latency": "high",
            "expected_cost": "high",
            "uses_adk_agent": True,
            "uses_reviewer": True,
        },
    }[selected_mode]
    return RouteDecision(
        requested_mode=requested_mode,
        selected_mode=selected_mode,
        label=RUN_MODE_LABELS[selected_mode],
        reason=reason,
        accuracy_priority=accuracy_priority,
        latency_priority=latency_priority,
        expected_latency=str(metadata["expected_latency"]),
        expected_cost=str(metadata["expected_cost"]),
        uses_retrieval=True,
        uses_adk_agent=bool(metadata["uses_adk_agent"]),
        uses_reviewer=bool(metadata["uses_reviewer"]),
    )


def _route_reason(
    requested_mode: str,
    selected_mode: str,
    *,
    query: str,
    accuracy: int,
    latency: int,
) -> str:
    if requested_mode != AUTO:
        return "User selected this route in the demo controls."
    if selected_mode == REVIEWED_AGENT:
        return "Auto chose Multi-agent because accuracy, source support, or policy sensitivity is high."
    if selected_mode == AGENTIC_RAG:
        return "Auto chose Agentic RAG because the query appears multi-step."
    return "Auto chose RAG because latency is prioritized and the query is narrow."


def _normalize_mode(value: str) -> str:
    mode = str(value or REVIEWED_AGENT).strip().lower()
    return mode if mode in SUPPORTED_RUN_MODES else REVIEWED_AGENT


def _bounded_priority(value: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = 3
    return min(5, max(1, numeric))
