from __future__ import annotations

from typing import Any

from defence_agent.evals.graders.common import contains_all, lower_answer, pass_fail, tool_names
from defence_agent.evals.schemas import EvalCase


EXPECTED_OVERDUE_TERMS = [
    "PB-SOP-2025",
    "52 days overdue",
    "EC-PROC-2025",
    "35 days overdue",
    "EC-PROC-2025-FR",
    "LOG-RET-2025",
    "96 days overdue",
]


def grade_code_execution(case: EvalCase, response: dict[str, Any], trace: dict[str, Any] | None = None) -> dict[str, Any]:
    if case.expected_route != "structured_table_analysis":
        return pass_fail(True, "CODE_EXECUTION_WRONG", sandbox_required=False)
    actual_tools = tool_names(response)
    sandbox_used = "run_table_analysis" in actual_tools
    answer = lower_answer(response)
    lowered_query = case.user_query.lower()
    if "overdue" in lowered_query:
        exact_terms = EXPECTED_OVERDUE_TERMS
    elif "count approved" in lowered_query and "owner" in lowered_query:
        exact_terms = ["Planning Policy Directorate", "Emergency Coordination Office", "Records Management Office"]
    else:
        exact_terms = ["doctrine_review_tracker"]
    exact_ok = contains_all(answer, exact_terms)
    sandbox_policy = _sandbox_policy(trace)
    policy_ok = bool(sandbox_policy) and sandbox_policy.get("network") == "disabled"
    row_ids_preserved = bool(_sandbox_row_ids(trace) or any(citation.get("row_id") for citation in response.get("citations", [])))
    passed = sandbox_used and exact_ok and policy_ok and row_ids_preserved
    return pass_fail(
        passed,
        "CODE_EXECUTION_WRONG",
        sandbox_used=sandbox_used,
        exact_result_correct=exact_ok,
        policy_visible=bool(sandbox_policy),
        network_disabled=policy_ok,
        row_ids_preserved=row_ids_preserved,
    )


def _sandbox_policy(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trace:
        return None
    for span in trace.get("spans", []):
        if span.get("name") == "sandbox_policy_checked":
            attrs = span.get("attributes") or {}
            if isinstance(attrs, dict):
                return attrs.get("policy")
    return None


def _sandbox_row_ids(trace: dict[str, Any] | None) -> list[str]:
    if not trace:
        return []
    for span in trace.get("spans", []):
        if span.get("name") == "sandbox_policy_checked":
            attrs = span.get("attributes") or {}
            if isinstance(attrs, dict):
                return attrs.get("row_ids") or []
    return []
