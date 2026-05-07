from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import pass_fail, tool_names
from defence_agent.evals.schemas import EvalCase


def grade_tools(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    actual = tool_names(response)
    expected = set(case.expected_tools)
    disallowed = set(case.disallowed_tools)
    missing = sorted(expected - set(actual))
    unexpected = sorted(disallowed.intersection(actual))
    errors = [call for call in response.get("tool_calls", []) if isinstance(call, dict) and call.get("ok") is False]
    passed = not missing and not unexpected and not errors
    return pass_fail(
        passed,
        "TOOL_ERROR" if errors else "TOOL_WRONG",
        expected_tools=sorted(expected),
        actual_tools=actual,
        missing_tools=missing,
        unexpected_tools=unexpected,
        tool_errors=len(errors),
        tool_call_count=len(actual),
    )
