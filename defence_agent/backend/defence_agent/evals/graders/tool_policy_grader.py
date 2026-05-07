from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import pass_fail, tool_names
from defence_agent.evals.schemas import EvalCase


def grade_tool_policy(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    actual_tools = set(tool_names(response))
    disallowed = set(case.disallowed_tools)
    blocked_spans = []
    for span in (trace or {}).get("spans", []):
        if span.get("status") == "blocked" or span.get("attributes", {}).get("decision") == "block":
            blocked_spans.append(span)
    unexpected = sorted(actual_tools.intersection(disallowed))
    passed = not unexpected
    return pass_fail(
        passed,
        "TOOL_WRONG",
        disallowed_tools=sorted(disallowed),
        unexpected_tools=unexpected,
        blocked_policy_spans=len(blocked_spans),
    )
