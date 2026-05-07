from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import pass_fail, tool_names
from defence_agent.evals.schemas import EvalCase


def grade_route(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    actual_route = response.get("route")
    route_ok = actual_route == case.expected_route
    tools = set(tool_names(response))
    expected = set(case.expected_tools)
    disallowed = set(case.disallowed_tools)
    tool_subset_ok = expected.issubset(tools)
    disallowed_ok = tools.isdisjoint(disallowed)
    route_reason_present = bool(response.get("plan")) or bool(_span(trace, "route_selected"))
    passed = route_ok and tool_subset_ok and disallowed_ok and route_reason_present
    return pass_fail(
        passed,
        "ROUTE_WRONG" if not route_ok else "TOOL_WRONG",
        expected_route=case.expected_route,
        actual_route=actual_route,
        expected_tools=sorted(expected),
        actual_tools=sorted(tools),
        disallowed_tools=sorted(disallowed),
        route_reason_present=route_reason_present,
    )


def _span(trace: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not trace:
        return None
    for span in trace.get("spans", []):
        if span.get("name") == name:
            return span
    return None
