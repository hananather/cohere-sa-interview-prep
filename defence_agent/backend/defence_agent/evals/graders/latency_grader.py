from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import pass_fail
from defence_agent.evals.schemas import EvalCase


def grade_operations(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    latency = response.get("latency_ms") or (trace or {}).get("duration_ms")
    trace_id = response.get("trace_id") or (trace or {}).get("trace_id")
    token_estimate = response.get("token_cost_estimate", {})
    passed = bool(trace_id) and latency is not None
    return pass_fail(
        passed,
        "TRACE_MISSING",
        latency_ms=latency,
        trace_id=trace_id,
        token_estimate_present=bool(token_estimate),
        token_estimate=token_estimate,
    )
