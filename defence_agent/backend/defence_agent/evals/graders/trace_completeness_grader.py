from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import pass_fail
from defence_agent.evals.schemas import EvalCase


def grade_trace_completeness(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    trace = trace or {}
    span_names = {span.get("name") for span in trace.get("spans", [])}
    required = {"input_safety_checked", "route_selected", "plan_created", "citation_validation_completed", "output_safety_checked"}
    if case.expected_tools:
        required.add("before_tool_policy")
        required.add("tool_call_started")
    missing = sorted(required - span_names)
    passed = bool(trace.get("trace_id")) and not missing
    return pass_fail(
        passed,
        "TRACE_MISSING",
        span_count=len(trace.get("spans", [])),
        missing_spans=missing,
        required_spans=sorted(required),
    )
